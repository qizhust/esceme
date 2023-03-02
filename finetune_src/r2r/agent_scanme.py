import sys
import networkx as nx
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F

from .eval_utils import cal_dtw
from utils.misc import length2mask
from .agent_tdstp import TDSTPAgent
from models.model_TDSTP import VLNBertScan, Critic


class ScanmeAgent(TDSTPAgent):
    def _build_model(self):
        self.vln_bert = VLNBertScan(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        self.scan_graphs = {}
        self.sg_vp2idx = defaultdict(dict)

    def _max_pool(self, feat_mat):
        pooled, _ = torch.max(feat_mat, 0)
        return pooled

    def _mean_pool(self, feat_mat):
        return feat_mat.mean(0)

    def _update_graph(self, obs, ended, a_t, max_hist_len, ob_img_feats, ob_nav_types):
        for i, ob in enumerate(obs):
            if ended[i]:
                self.seq_dup_vp[i].append(True)
                continue
            vp = ob['viewpoint']
            if vp not in self.vp2idx_list[i]:
                idx = len(self.vp2idx_list[i])
                self.vp2idx_list[i][vp] = idx
                self.graphs[i].add_node(vp)
                self.seq_dup_vp[i].append(False)
                self.seq_last_idx[i][vp] = len(self.seq_dup_vp[i]) - 1
            else:
                idx = self.vp2idx_list[i][vp]
                if self.args.no_temporal_strategy == 'replace':
                    self.seq_dup_vp[i].append(False)
                    self.seq_dup_vp[i][self.seq_last_idx[i][vp]] = True
                else:  # 'keep' strategy, keep the old one
                    self.seq_dup_vp[i].append(True)
                self.seq_last_idx[i][vp] = len(self.seq_dup_vp[i]) - 1
            self.seq_idx_list[i].append(idx)
            self.seq_vp_list[i].append(vp)
            self.seq_view_idx_list[i].append(ob['viewIndex'])
            self.seq_dist_list[i].append(ob['distance'])

            if ob['scan'] not in self.scan_graphs:
                self.scan_graphs[ob['scan']] = nx.Graph()
            if vp not in self.scan_graphs[ob['scan']]:
                pooled_feat = getattr(self, '_'+self.args.pooling+'_pool')(ob_img_feats[i][ob_nav_types[i]==1].cpu())
                self.scan_graphs[ob['scan']].add_node(vp, feat=pooled_feat)
                self.sg_vp2idx[ob['scan']][vp] = len(self.sg_vp2idx[ob['scan']])

            for adj in ob['candidate']:
                adj_vp = adj['viewpointId']
                if adj_vp in self.vp2idx_list[i]:
                    self.graphs[i].add_edge(vp, adj_vp)

                if adj_vp in self.scan_graphs[ob['scan']]:
                    self.scan_graphs[ob['scan']].add_edge(vp, adj_vp)

            # block path if backtrack
            if max_hist_len > a_t[i] >= 0:
                hist_vp = self.seq_vp_list[i][a_t[i] - 1]
                self.blocked_path[i][hist_vp][vp] += 1

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types, ob_pos, ob_node_fts = [], [], [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types, cand_pos, cand_node_fts = [], [], [], [], []
            cand_pointids = np.zeros((self.args.views, ), dtype=np.bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
                if self.args.cand_use_ob_pos:
                    cand_pos.append(ob['position'])
                else:
                    cand_pos.append(cc['position'])
                if cc['viewpointId'] in self.scan_graphs.get(ob['scan'], {}):
                    cand_node_fts.append(self.scan_graphs[ob['scan']].nodes[cc['viewpointId']]['feat'].unsqueeze(0))
                else:
                    cand_node_fts.append(torch.zeros_like(torch.from_numpy(cand_img_fts[-1])).unsqueeze(0))
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size, ), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size, ), dtype=np.float32))
            cand_pos.append(ob['position'])
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)
            cand_node_fts.append(torch.zeros((1, self.args.image_feat_size)))
            cand_node_fts = torch.cat(cand_node_fts, 0)
            
            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))
            cand_pos.extend([ob['position'] for _ in range(self.args.views - np.sum(cand_pointids))])
            cand_pano_node_fts = torch.cat([cand_node_fts, torch.from_numpy(pano_fts[:, :self.args.image_feat_size])], 0)  # redundant pano_fts, just for alignment

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)
            ob_node_fts.append(cand_pano_node_fts)
            ob_pos.append(cand_pos)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)
            ob_pos[i] = np.array(ob_pos[i] + [np.array([0, 0, 0], dtype=np.float32) for _ in range(num_pads)])
            ob_node_fts[i] = torch.cat([ob_node_fts[i], torch.zeros((num_pads, ob_node_fts[i].shape[1]))], 0).unsqueeze(0)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()
        ob_node_fts = torch.cat(ob_node_fts, 0).cuda()
        ob_pos = torch.from_numpy(np.stack(ob_pos, 0)).float().cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens, ob_node_fts, ob_pos

    def _candidate_variable(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        cand_node_fts = torch.zeros((len(obs), max_len, self.args.image_feat_size))
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
                if cc['viewpointId'] in self.scan_graphs.get(ob['scan'], {}):
                    cand_node_fts[i, j] = self.scan_graphs[ob['scan']].nodes[cc['viewpointId']]['feat']
                
            cand_nav_types[i, cand_lens[i]-1] = 2

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens, cand_node_fts.cuda()

    def _get_pano_input(self, obs):
        ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_node_feats, ob_pos = self._cand_pano_feature_variable(obs)
        ob_masks = length2mask(ob_lens).logical_not()
        return ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats, ob_pos

    def _get_cand_input(self, obs):
        ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_node_feats = self._candidate_variable(obs)
        ob_masks = length2mask(ob_cand_lens).logical_not()
        return ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats, None

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        rl_teacher_loss = 0.
        target_predict_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        base_position = np.stack([ob['position'] for ob in obs], axis=0)

        goal_positions = np.stack([ob['goal_position'] for ob in obs], axis=0) - base_position[:, :2]  # bs x 2
        global_positions, goal_distances = self._init_global_positions(obs, goal_positions)  # [B, 5**2, 3]
        global_pos_feat = self.vln_bert('global_pos', global_pos=global_positions, txt_embeds=txt_embeds)  # [B, 5**2, D]
        goal_pred_gt = torch.argmin(goal_distances, dim=1)  # [B,]

        # global embedding
        hist_embeds = self.vln_bert('history').expand(batch_size, -1).unsqueeze(1)  # [B, 1, D]
        hist_lens = [1 for _ in range(batch_size)]

        self._init_graph(batch_size)  # init the graphs for the current batch

        for t in range(self.args.max_action_len):
            ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats, ob_pos = getattr(self, '_get_'+self.args.ob_type+'_input')(obs)

            ''' Visual BERT '''
            graph_mask = self._get_connectivity_mask() if t > 0 and self.args.use_conn else None
            vp_dup = torch.Tensor(self.seq_dup_vp).bool().cuda() if self.args.no_temporal and t > 0 else None
            ob_pos = ob_pos - torch.from_numpy(base_position).cuda().unsqueeze(1)  # bs x num_obs x 3
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,  # history before t step
                'hist_lens': hist_lens,
                'graph_mask': graph_mask,
                'vp_dup': vp_dup,
                'ob_node_feats': ob_node_feats,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'ob_position': ob_pos.float(),
                'return_states': True if self.feedback == 'sample' else False,
                'global_pos_feat': global_pos_feat if self.args.global_positions else None
            }

            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            hist_embeds = t_outputs[-1]
            if self.args.global_positions:
                pos_logit = t_outputs[-3]
                global_pos_feat = t_outputs[-2]
            max_hist_len = hist_embeds.size(1)  # max history length
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            # mask out logits of the current position
            if t > 0:
                logit[:, 1:max_hist_len].masked_fill_(self._get_dup_logit_mask(obs) == 0, -float('inf'))
            if self.args.no_gas:
                logit[:, 1:max_hist_len].fill_(-float('inf'))

            # reweight logits
            if self.args.logit_reweighting and not self.args.no_gas:
                for idx, ob in enumerate(obs):
                    current_vp = ob['viewpoint']
                    for jdx, hist_vp in enumerate(self.seq_vp_list[idx]):
                        logit[idx, 1 + jdx] -= self.blocked_path[idx][current_vp][hist_vp] * 1e2
                    for jdx, cand in enumerate(ob['candidate']):
                        logit[idx, max_hist_len + jdx] -= self.blocked_path[idx][current_vp][cand['viewpointId']] * 1e2

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(obs, ended, max_hist_len)
                ml_loss += self.criterion(logit, target)
            if train_rl and not self.args.no_gas:
                rl_forbidden = (logit < -1e7)
                rl_target = self._get_rl_optimal_action(obs, ended, max_hist_len, rl_forbidden)
                rl_teacher_loss += self.criterion(logit, rl_target)

            if self.args.global_positions and (train_ml is not None or train_rl):
                goal_pred_gt = goal_pred_gt.clone().detach()
                for i, end in enumerate(ended):
                    if end:
                        goal_pred_gt[i] = -100
                target_predict_loss += self.criterion(pos_logit, goal_pred_gt)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:  # default: skip
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit[:, max_hist_len:].masked_fill_(bt_masks, -float('inf'))


            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit / self.args.rl_temperature, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id - max_hist_len == (ob_cand_lens[i] - 1) or next_id == self.args.ignoreid or \
                        ended[i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            self._update_graph(obs, ended, cpu_a_t, max_hist_len, ob_img_feats, ob_nav_types)
            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len - 1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        if next_id >= max_hist_len:
                            prev_act_angle[i] = \
                                obs[i]['candidate'][next_id - max_hist_len]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                position = np.stack([ob['position'] for ob in obs], axis=0) - base_position  # bs x 3
                position = torch.from_numpy(position).cuda().float()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t if not self.args.no_temporal else None,
                    'position': position,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds = torch.cat([hist_embeds, t_hist_embeds.unsqueeze(1)], dim=1)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, max_hist_len, traj=traj)
            obs = self.env._get_obs(t=t+1)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats, ob_pos = getattr(self, '_get_'+self.args.ob_type+'_input')(obs)

            ''' Visual BERT '''
            graph_mask = self._get_connectivity_mask() if self.args.use_conn else None
            ob_pos = ob_pos - torch.from_numpy(base_position).cuda().unsqueeze(1)
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'graph_mask': graph_mask,
                'ob_node_feats': ob_node_feats,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'ob_position': ob_pos.float(),
                'return_states': True,
                'global_pos_feat': global_pos_feat if self.args.global_positions else None
            }
            temp_output = self.vln_bert(**visual_inputs)
            last_h_ = temp_output[1]

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            if self.args.rl_teacher_only:
                rl_loss = rl_teacher_loss * self.args.rl_teacher_weight / batch_size
            else:
                rl_loss += rl_teacher_loss * self.args.rl_teacher_weight / batch_size
            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())  # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if self.args.global_positions and (train_rl or train_ml is not None):
            self.loss += target_predict_loss * self.args.gp_loss_weight / batch_size
            self.logs['GP_loss'].append((target_predict_loss * self.args.gp_loss_weight / batch_size).item())
        elif not self.args.global_positions:
            self.logs['GP_loss'].append(0)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj
