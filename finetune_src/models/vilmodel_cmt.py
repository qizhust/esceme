import logging
import math
import copy

import torch
from torch import nn

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.no_lang_ca = config.no_lang_ca # do not update language embeds

        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        if self.no_lang_ca:
            lang_att_output = lang_input
        else:
            lang_att_output, _ = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output, _ = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        if self.no_lang_ca:
            lang_att_output = (lang_input, )
        else:
            lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        if not self.no_lang_ca:
            lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        if self.no_lang_ca:
            lang_output = lang_input
        else:
            lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask, visn_self_attn_mask=None):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        if visn_self_attn_mask is None:
            visn_self_attn_mask = visn_attention_mask
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output[0], visn_att_output[0])

        return lang_output, visn_output

class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_l_layers = config.num_l_layers
        self.num_r_layers = config.num_r_layers
        self.num_h_layers = config.num_h_layers
        self.num_x_layers = config.num_x_layers
        self.update_lang_bert = config.update_lang_bert

        # Using self.layer instead of self.l_layers to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

        self.h_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, extended_txt_masks, hist_embeds,
                extended_hist_masks, img_embeds=None, extended_img_masks=None):
        # text encoding
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()

        # image encoding
        if img_embeds is not None:
            if self.r_layers is not None:
                for layer_module in self.r_layers:
                    temp_output = layer_module(img_embeds, extended_img_masks)
                    img_embeds = temp_output[0]

        # history encoding
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        hist_max_len = hist_embeds.size(1)
        
        # cross-modal encoding
        if img_embeds is None:
            hist_img_embeds = hist_embeds
            extended_hist_img_masks = extended_hist_masks
        else:
            hist_img_embeds = torch.cat([hist_embeds, img_embeds], 1)
            extended_hist_img_masks = torch.cat([extended_hist_masks, extended_img_masks], -1)
        
        for layer_module in self.x_layers:
            txt_embeds, hist_img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                hist_img_embeds, extended_hist_img_masks)

        hist_embeds = hist_img_embeds[:, :hist_max_len]
        if img_embeds is not None:
            img_embeds = hist_img_embeds[:, hist_max_len:]
        return txt_embeds, hist_embeds, img_embeds



class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, ang_feat, type_embeddings, nav_types=None):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        embeddings = transformed_im + transformed_ang + type_embeddings
        if nav_types is not None:
            nav_embeddings = self.nav_type_embedding(nav_types)
            embeddings = embeddings + nav_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class HistoryEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        self.position_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        # special type embedding for history
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hist_enc_pano = config.hist_enc_pano
        if config.hist_enc_pano:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_enc_config)
        else:
            self.pano_encoder = None

    def forward(self, img_feats, ang_feats, pos_ids, 
                pano_img_feats=None, pano_ang_feats=None):
        '''Args:
        - img_feats: (batch_size, dim_feat)
        - pos_ids: (batch_size, )
        - pano_img_feats: (batch_size, pano_len, dim_feat)
        '''
        device = next(iter(self.parameters())).device
        if img_feats is not None:
            batch_size = img_feats.size(0)
        else:
            batch_size = 1

        type_ids = torch.zeros((batch_size, )).long().to(device)
        type_embeddings = self.type_embedding(type_ids)

        if img_feats is None:
            cls_embeddings = self.dropout(self.layer_norm(
                self.cls_token.expand(batch_size, -1, -1)[:, 0] + type_embeddings))
            return cls_embeddings

        # history embedding per step
        embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                     self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                     self.position_embeddings(pos_ids) + \
                     type_embeddings

        if self.pano_encoder is not None:
            pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                              self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
            pano_embeddings = self.dropout(pano_embeddings)
            # TODO: mask is always True
            batch_size, pano_len, _ = pano_img_feats.size()
            extended_pano_masks = torch.zeros(batch_size, pano_len).float().to(device).unsqueeze(1).unsqueeze(2)
            pano_embeddings = self.pano_encoder(pano_embeddings, extended_pano_masks)[0]
            pano_embeddings = torch.mean(pano_embeddings, 1)

            embeddings = embeddings + pano_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class NavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)
        
        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.init_weights()

    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
                ob_masks=None):
        
        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = torch.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            for layer_module in self.encoder.layer:
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            if self.config.fix_lang_embedding:
                txt_embeds = txt_embeds.detach()
            if self.config.no_lang_ca: # run self-attn layers for lang
                all_txt_embeds = [txt_embeds]
                for layer_module in self.encoder.x_layers:
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                pano_img_feats=hist_pano_img_feats, pano_ang_feats=hist_pano_ang_feats)
            if self.config.fix_hist_embedding:
                hist_embeds = hist_embeds.detach()
            return hist_embeds
            
        # cross-modal encoding per step
        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(hist_embeds, extended_hist_masks)
                    hist_embeds = temp_output[0]

            # image embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0

            ob_token_type_ids = torch.ones(ob_img_feats.size(0), ob_img_feats.size(1), dtype=torch.long, device=self.device)
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats, 
                self.embeddings.token_type_embeddings(ob_token_type_ids), 
                nav_types=ob_nav_types)
            if self.encoder.r_layers is not None:
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0]
            if self.config.fix_obs_embedding:
                ob_embeds = ob_embeds.detach()

            # multi-modal encoding
            hist_max_len = hist_embeds.size(1)
            hist_ob_embeds = torch.cat([hist_embeds, ob_embeds], 1)
            extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks], -1)

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0
            
            if self.config.no_lang_ca:
                all_txt_embeds = txt_embeds
            for l, layer_module in enumerate(self.encoder.x_layers):
                if self.config.no_lang_ca:
                    txt_embeds = all_txt_embeds[l]
                txt_embeds, hist_ob_embeds = layer_module(
                    txt_embeds, extended_txt_masks, 
                    hist_ob_embeds, extended_hist_ob_masks,
                )

            hist_embeds = hist_ob_embeds[:, :hist_max_len]
            ob_embeds = hist_ob_embeds[:, hist_max_len:]

            # TODO
            if self.config.no_lang_ca:
                act_logits = self.next_action(ob_embeds).squeeze(-1)
            else:
                if self.config.act_pred_token == 'ob_txt':
                    act_logits = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    act_logits = self.next_action(ob_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(ob_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    act_logits = self.next_action(ob_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1)
            act_logits.masked_fill_(ob_nav_types==0, -float('inf'))

            return act_logits, txt_embeds, hist_embeds, ob_embeds


class NavHAMTScanme(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)
        
        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.ob_node_fusion = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(config.hidden_size, eps=1e-12),
                                 nn.Linear(config.hidden_size, config.hidden_size))

        self.init_weights()

    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
                ob_masks=None, ob_node_feats=None):
        
        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = torch.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            for layer_module in self.encoder.layer:
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            if self.config.fix_lang_embedding:
                txt_embeds = txt_embeds.detach()
            if self.config.no_lang_ca: # run self-attn layers for lang
                all_txt_embeds = [txt_embeds]
                for layer_module in self.encoder.x_layers:
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                pano_img_feats=hist_pano_img_feats, pano_ang_feats=hist_pano_ang_feats)
            if self.config.fix_hist_embedding:
                hist_embeds = hist_embeds.detach()
            return hist_embeds
            
        # cross-modal encoding per step
        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(hist_embeds, extended_hist_masks)
                    hist_embeds = temp_output[0]

            # image embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0

            ob_token_type_ids = torch.ones(ob_img_feats.size(0), ob_img_feats.size(1), dtype=torch.long, device=self.device)
            ob_img_node_feats = self.ob_node_fusion(torch.cat([ob_img_feats, ob_node_feats], -1))
            ob_embeds = self.img_embeddings(ob_img_node_feats, ob_ang_feats, 
                self.embeddings.token_type_embeddings(ob_token_type_ids), 
                nav_types=ob_nav_types)
            if self.encoder.r_layers is not None:
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0]
            if self.config.fix_obs_embedding:
                ob_embeds = ob_embeds.detach()

            # multi-modal encoding
            hist_max_len = hist_embeds.size(1)
            ob_len = ob_embeds.size(1)
            hist_ob_embeds = torch.cat([hist_embeds, ob_embeds], 1)
            extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks], -1)

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0
            
            if self.config.no_lang_ca:
                all_txt_embeds = txt_embeds
            for l, layer_module in enumerate(self.encoder.x_layers):
                if self.config.no_lang_ca:
                    txt_embeds = all_txt_embeds[l]
                txt_embeds, hist_ob_embeds = layer_module(
                    txt_embeds, extended_txt_masks, 
                    hist_ob_embeds, extended_hist_ob_masks,
                )

            hist_embeds = hist_ob_embeds[:, :hist_max_len]
            ob_embeds = hist_ob_embeds[:, hist_max_len:hist_max_len+ob_len]

            # TODO
            if self.config.no_lang_ca:
                act_logits = self.next_action(ob_embeds).squeeze(-1)
            else:
                if self.config.act_pred_token == 'ob_txt':
                    act_logits = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    act_logits = self.next_action(ob_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(ob_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    act_logits = self.next_action(ob_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1)
            act_logits.masked_fill_(ob_nav_types==0, -float('inf'))

            return act_logits, txt_embeds, hist_embeds, ob_embeds


class NavTDSTP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)

        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.init_weights()

    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None,
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None, graph_mask=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
                ob_masks=None, history_mapper=None, global_pos_feat=None, ob_position_feat=None):

        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = torch.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            for layer_module in self.encoder.layer:
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            if self.config.fix_lang_embedding:
                txt_embeds = txt_embeds.detach()
            if self.config.no_lang_ca:  # run self-attn layers for lang
                all_txt_embeds = [txt_embeds]
                for layer_module in self.encoder.x_layers:
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                                               pano_img_feats=hist_pano_img_feats, pano_ang_feats=hist_pano_ang_feats)
            if self.config.fix_hist_embedding:
                hist_embeds = hist_embeds.detach()
            return hist_embeds

        # cross-modal encoding per step
        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)

            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(hist_embeds, extended_hist_masks)
                    hist_embeds = temp_output[0]

            # image embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)

            ob_token_type_ids = torch.ones(ob_img_feats.size(0), ob_img_feats.size(1), dtype=torch.long,
                                           device=self.device)
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats,
                                            self.embeddings.token_type_embeddings(ob_token_type_ids),
                                            nav_types=ob_nav_types)
            ob_embeds += ob_position_feat

            if self.encoder.r_layers is not None:
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0]
            if self.config.fix_obs_embedding:
                ob_embeds = ob_embeds.detach()

            # multi-modal encoding
            hist_max_len = hist_embeds.size(1)
            ob_len = ob_embeds.size(1)
            if global_pos_feat is not None:
                global_pos_embeds = global_pos_feat
                global_pos_mask = torch.ones((global_pos_embeds.size(0), global_pos_embeds.size(1)))\
                    .unsqueeze(1).unsqueeze(2).cuda()
                hist_ob_embeds = torch.cat([hist_embeds, ob_embeds, global_pos_embeds], 1)
                extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks, global_pos_mask], -1)
            else:
                hist_ob_embeds = torch.cat([hist_embeds, ob_embeds], 1)
                extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks], -1)
            extended_hist_ob_masks_graph = extended_hist_ob_masks.transpose(-1, -2) * extended_hist_ob_masks
            if graph_mask is not None:
                graph_max_size = graph_mask.size(-1)
                extended_hist_ob_masks_graph[:, 0, 1:graph_max_size+1, 1:graph_max_size+1] *= graph_mask
            extended_hist_ob_masks = (1.0 - extended_hist_ob_masks) * -10000.0
            extended_hist_ob_masks_graph = (1.0 - extended_hist_ob_masks_graph) * -10000.0

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            if self.config.no_lang_ca:
                all_txt_embeds = txt_embeds
            for l, layer_module in enumerate(self.encoder.x_layers):
                if self.config.no_lang_ca:
                    txt_embeds = all_txt_embeds[l]
                txt_embeds, hist_ob_embeds = layer_module(
                    txt_embeds, extended_txt_masks,
                    hist_ob_embeds, extended_hist_ob_masks, visn_self_attn_mask=extended_hist_ob_masks_graph
                )

            hist_embeds = hist_ob_embeds[:, :hist_max_len]
            ob_embeds = hist_ob_embeds[:, hist_max_len:hist_max_len+ob_len]
            if global_pos_feat is not None:
                pos_embeds = hist_ob_embeds[:, hist_max_len+ob_len:]
            if history_mapper is not None:
                hist_cand_embeds = history_mapper(hist_embeds)
            else:
                hist_cand_embeds = hist_embeds
            ob_cand_embeds = ob_embeds
            cand_embeds = torch.cat([hist_cand_embeds, ob_cand_embeds], dim=1)

            # TODO
            if self.config.no_lang_ca:
                act_logits = self.next_action(cand_embeds).squeeze(-1)
            else:
                if self.config.act_pred_token == 'ob_txt':  # default
                    act_logits = self.next_action(cand_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    act_logits = self.next_action(cand_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(cand_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    act_logits = self.next_action(cand_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1)
            act_logits[:, hist_max_len:].masked_fill_(ob_nav_types == 0, -float('inf'))
            act_logits[:, :hist_max_len].masked_fill_(hist_masks == False, -float('inf'))
            act_logits[:, 0] = -float('inf')

            if global_pos_feat is not None:
                return act_logits, txt_embeds, hist_embeds, ob_embeds, pos_embeds
            else:
                return act_logits, txt_embeds, hist_embeds, ob_embeds


class NavScanme(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)

        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.ob_node_fusion = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(config.hidden_size, eps=1e-12),
                                 nn.Linear(config.hidden_size, config.hidden_size))

        self.init_weights()

    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None,
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None, graph_mask=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
                ob_masks=None, history_mapper=None, global_pos_feat=None, ob_position_feat=None, ob_node_feats=None):

        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = torch.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            for layer_module in self.encoder.layer:
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            if self.config.fix_lang_embedding:
                txt_embeds = txt_embeds.detach()
            if self.config.no_lang_ca:  # run self-attn layers for lang
                all_txt_embeds = [txt_embeds]
                for layer_module in self.encoder.x_layers:
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                                               pano_img_feats=hist_pano_img_feats, pano_ang_feats=hist_pano_ang_feats)
            if self.config.fix_hist_embedding:
                hist_embeds = hist_embeds.detach()
            return hist_embeds

        # cross-modal encoding per step
        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)

            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(hist_embeds, extended_hist_masks)
                    hist_embeds = temp_output[0]

            # image embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)

            ob_token_type_ids = torch.ones(ob_img_feats.size(0), ob_img_feats.size(1), dtype=torch.long,
                                           device=self.device)
            ob_img_node_feats = self.ob_node_fusion(torch.cat([ob_img_feats, ob_node_feats], -1))
            ob_embeds = self.img_embeddings(ob_img_node_feats, ob_ang_feats,
                                            self.embeddings.token_type_embeddings(ob_token_type_ids),
                                            nav_types=ob_nav_types)
            ob_embeds += ob_position_feat

            if self.encoder.r_layers is not None:
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0]
            if self.config.fix_obs_embedding:
                ob_embeds = ob_embeds.detach()

            # multi-modal encoding
            hist_max_len = hist_embeds.size(1)
            ob_len = ob_embeds.size(1)
            if global_pos_feat is not None:
                global_pos_embeds = global_pos_feat
                global_pos_mask = torch.ones((global_pos_embeds.size(0), global_pos_embeds.size(1)))\
                    .unsqueeze(1).unsqueeze(2).cuda()
                hist_ob_embeds = torch.cat([hist_embeds, ob_embeds, global_pos_embeds], 1)
                extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks, global_pos_mask], -1)
            else:
                hist_ob_embeds = torch.cat([hist_embeds, ob_embeds], 1)
                extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks], -1)

            extended_hist_ob_masks_graph = extended_hist_ob_masks.transpose(-1, -2) * extended_hist_ob_masks
            if graph_mask is not None:
                graph_max_size = graph_mask.size(-1)
                extended_hist_ob_masks_graph[:, 0, 1:graph_max_size+1, 1:graph_max_size+1] *= graph_mask
            extended_hist_ob_masks = (1.0 - extended_hist_ob_masks) * -10000.0
            extended_hist_ob_masks_graph = (1.0 - extended_hist_ob_masks_graph) * -10000.0

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            if self.config.no_lang_ca:
                all_txt_embeds = txt_embeds
            for l, layer_module in enumerate(self.encoder.x_layers):
                if self.config.no_lang_ca:
                    txt_embeds = all_txt_embeds[l]
                txt_embeds, hist_ob_embeds = layer_module(
                    txt_embeds, extended_txt_masks,
                    hist_ob_embeds, extended_hist_ob_masks, visn_self_attn_mask=extended_hist_ob_masks_graph
                )

            hist_embeds = hist_ob_embeds[:, :hist_max_len]
            ob_embeds = hist_ob_embeds[:, hist_max_len:hist_max_len+ob_len]
            if global_pos_feat is not None:
                pos_embeds = hist_ob_embeds[:, hist_max_len+ob_len:]
            if history_mapper is not None:
                hist_cand_embeds = history_mapper(hist_embeds)
            else:
                hist_cand_embeds = hist_embeds
            ob_cand_embeds = ob_embeds
            cand_embeds = torch.cat([hist_cand_embeds, ob_cand_embeds], dim=1)

            # TODO
            if self.config.no_lang_ca:
                act_logits = self.next_action(cand_embeds).squeeze(-1)
            else:
                if self.config.act_pred_token == 'ob_txt':  # default
                    act_logits = self.next_action(cand_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    act_logits = self.next_action(cand_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(cand_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    act_logits = self.next_action(cand_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1)
            act_logits[:, hist_max_len:].masked_fill_(ob_nav_types == 0, -float('inf'))
            act_logits[:, :hist_max_len].masked_fill_(hist_masks == False, -float('inf'))
            act_logits[:, 0] = -float('inf')

            if global_pos_feat is not None:
                return act_logits, txt_embeds, hist_embeds, ob_embeds, pos_embeds
            else:
                return act_logits, txt_embeds, hist_embeds, ob_embeds


class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds
    
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings_duet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens
        
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds
       
    
class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = ImageEmbeddings_duet(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)
        if config.glocal_fuse:
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        else:
            self.sap_fuse_linear = None
        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.config.hidden_size)
        
        self.init_weights()
        
        if config.fix_lang_embedding or config.fix_local_branch:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding or config.fix_local_branch:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False
        if config.fix_local_branch:
            for k, v in self.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.local_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.og_head.named_parameters():
                v.requires_grad = False
    
    def forward_text(self, txt_ids, txt_masks):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds

    def forward_panorama_per_step(
        self, view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens
    ):
        device = view_img_fts.device
        has_obj = obj_img_fts is not None

        view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(view_img_fts)
        )
        if has_obj:
            if self.img_embeddings.obj_linear is None:
                obj_img_embeds = self.img_embeddings.img_layer_norm(
                    self.img_embeddings.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.img_embeddings.obj_layer_norm(
                    self.img_embeddings.obj_linear(obj_img_fts)
                )
            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                view_img_embeds, obj_img_embeds, view_lens, obj_lens
            ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            pano_lens = view_lens + obj_lens
        else:
            img_embeds = view_img_embeds
            pano_lens = view_lens

        pano_embeds = img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        pano_masks = gen_seq_masks(pano_lens)
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
        return pano_embeds, pano_masks

    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,
    ):
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
       
        # local branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
            
        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        # print(fuse_weights)

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        # print('global', torch.softmax(global_logits, 1)[0], global_logits[0])

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))
        # print('local', torch.softmax(local_logits, 1)[0], local_logits[0])

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
        # print('fused', torch.softmax(fused_logits, 1)[0], fused_logits[0])

        # object grounding logits
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
        }
        return outs

    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'])
            return txt_embeds

        elif mode == 'panorama':
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks

        elif mode == 'navigation':
             return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
            )


class NavDUETScanme(GlocalTextPathNavCMT):

    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids, sgraph_embeds=None
    ):
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )

        # local branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        if sgraph_embeds is not None:
            vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, 
                torch.cat([vp_embeds, sgraph_embeds.unsqueeze(1)], 1), 
                torch.cat([vp_masks, torch.ones((batch_size, 1), dtype=bool).cuda()], -1))[:, :vp_embeds.size(1)]
        else:
            vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
            
        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        # print(fuse_weights)

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        # print('global', torch.softmax(global_logits, 1)[0], global_logits[0])

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))
        # print('local', torch.softmax(local_logits, 1)[0], local_logits[0])

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
        # print('fused', torch.softmax(fused_logits, 1)[0], fused_logits[0])

        # object grounding logits
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
        }
        return outs

    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'])
            return txt_embeds

        elif mode == 'panorama':
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks

        elif mode == 'navigation':
             return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'], batch['sgraph_embeds']
            )
