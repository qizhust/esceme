import sys
import numpy as np
from collections import defaultdict
import networkx as nx
import torch
import torch.nn.functional as F

from utils.misc import length2mask
from .agent_cmt import Seq2SeqCMTAgent
from .eval_utils import cal_dtw
# from utils.logger import print_progress
from models.model_HAMT import VLNBertScan, Critic


class HamtScanmeAgent(Seq2SeqCMTAgent):

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

    def _update_graph(self, obs, ob_img_feats, ob_nav_types):
        for i, ob in enumerate(obs):
            vp = ob['viewpoint']
            if ob['scan'] not in self.scan_graphs:
                self.scan_graphs[ob['scan']] = nx.Graph()
            if vp not in self.scan_graphs[ob['scan']]:
                pooled_feat = getattr(self, '_'+self.args.pooling+'_pool')(ob_img_feats[i][ob_nav_types[i]==1].cpu())
                self.scan_graphs[ob['scan']].add_node(vp, feat=pooled_feat)
                self.sg_vp2idx[ob['scan']][vp] = len(self.sg_vp2idx[ob['scan']])

            for adj in ob['candidate']:
                adj_vp = adj['viewpointId']
                if adj_vp in self.scan_graphs[ob['scan']]:
                    self.scan_graphs[ob['scan']].add_edge(vp, adj_vp)

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types, ob_node_fts = [], [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types, cand_node_fts = [], [], [], []
            cand_pointids = np.zeros((self.args.views, ), dtype=np.bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
                if cc['viewpointId'] in self.scan_graphs.get(ob['scan'], {}):
                    cand_node_fts.append(self.scan_graphs[ob['scan']].nodes[cc['viewpointId']]['feat'].unsqueeze(0))
                else:
                    cand_node_fts.append(torch.zeros_like(torch.from_numpy(cand_img_fts[-1])).unsqueeze(0))
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size, ), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size, ), dtype=np.float32))
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
            cand_pano_node_fts = torch.cat([cand_node_fts, torch.from_numpy(pano_fts[:, :self.args.image_feat_size])], 0)  # redundant pano_fts, just for alignment

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)
            ob_node_fts.append(cand_pano_node_fts)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)
            ob_node_fts[i] = torch.cat([ob_node_fts[i], torch.zeros((num_pads, ob_node_fts[i].shape[1]))], 0).unsqueeze(0)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()
        ob_node_fts = torch.cat(ob_node_fts, 0).cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens, ob_node_fts

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
        ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_node_feats = self._cand_pano_feature_variable(obs)
        ob_masks = length2mask(ob_lens).logical_not()
        return ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats

    def _get_cand_input(self, obs):
        ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_node_feats = self._candidate_variable(obs)
        ob_masks = length2mask(ob_cand_lens).logical_not()
        return ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats

    def _get_visual_input(self, txt_embeds, txt_masks, hist_embeds, hist_lens, ob_img_feats, ob_ang_feats, 
        ob_nav_types, ob_masks, ob_node_feats, return_states=False):

        visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'ob_node_feats': ob_node_feats,
                'return_states': return_states
            }
        return visual_inputs

    def _get_rl_res(self, obs, traj, ended, cpu_a_t, last_dist, last_ndtw):
        batch_size = len(obs)
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
                if action_idx == -1:                              # If the action now is end
                    if dist[i] < 3.0:                             # Correct
                        reward[i] = 2.0 + ndtw_score[i] * 2.0
                    else:                                         # Incorrect
                        reward[i] = -2.0
                else:                                             # The action is not end
                    # Path fidelity rewards (distance & nDTW)
                    reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                    ndtw_reward = ndtw_score[i] - last_ndtw[i]
                    if reward[i] > 0.0:                           # Quantification
                        reward[i] = 1.0 + ndtw_reward
                    elif reward[i] < 0.0:
                        reward[i] = -1.0 + ndtw_reward
                    else:
                        raise NameError("The action doesn't change the move")
                    # Miss the target penalty
                    if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                        reward[i] -= (1.0 - last_dist[i]) * 2.0

        return reward, mask, dist, ndtw_score

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        # Record starting point
        traj = [{
            'instruction': ob['instruction'],
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        batch_size = len(obs)

        if self.feedback in ['teacher', 'argmax']:
            train_rl = False
            return_state = False
        else:  # train_rl=True
            rewards = []
            hidden_states = []
            masks = []
            return_state = True
            # Init the reward shaping
            last_dist = np.zeros(batch_size, np.float32)
            last_ndtw = np.zeros(batch_size, np.float32)
            for i, ob in enumerate(obs):   # The init distance from the view point to the target
                last_dist[i] = ob['distance']
                path_act = [vp[0] for vp in traj[i]['path']]
                last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        # Init the logs
        policy_log_probs = []
        entropys = []
        self.ml_loss = 0.

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]

        for t in range(self.args.max_action_len):
            ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats = getattr(self, '_get_'+self.args.ob_type+'_input')(obs)
            
            ''' Visual BERT '''
            visual_inputs = self._get_visual_input(txt_embeds, txt_masks, hist_embeds, hist_lens, ob_img_feats, ob_ang_feats,
                ob_nav_types, ob_masks, ob_node_feats, return_state)

            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(obs, ended)
                self.ml_loss += self.criterion(logit, target)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            self._update_graph(obs, ob_img_feats, ob_nav_types)
            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs(t=t+1)

            if train_rl:
                reward, mask, dist, ndtw_score = self._get_rl_res(obs, traj, ended, cpu_a_t, last_dist, last_ndtw)
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens, ob_masks, ob_node_feats = getattr(self, '_get_'+self.args.ob_type+'_input')(obs)

            ''' Visual BERT '''
            visual_inputs = self._get_visual_input(txt_embeds, txt_masks, hist_embeds, hist_lens, ob_img_feats, ob_ang_feats,
                ob_nav_types, ob_masks, ob_node_feats, return_state)
            _, last_h_ = self.vln_bert(**visual_inputs)

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            rl_loss = 0.
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = -policy_log_probs[t] * a_ * mask_
                t_critic_loss = (((r_ - v_) ** 2) * mask_) * 0.5 # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss  # [B,]
                rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_)

                self.logs['critic_loss'].append(t_critic_loss.sum().item())
                self.logs['policy_loss'].append(t_policy_loss.sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            div_dict = {'total': total, 'batch': batch_size}
            rl_loss = rl_loss / div_dict[self.args.normalize_loss]
            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.sum().item()) # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += self.ml_loss * train_ml / batch_size  # [B,]
            self.logs['IL_loss'].append((self.ml_loss.sum() * train_ml / batch_size).item())

        return traj
