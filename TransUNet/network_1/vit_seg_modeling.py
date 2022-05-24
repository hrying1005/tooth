# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, InstanceNorm1d
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        # attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        # attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config

        self.patch_embeddings = Conv2d(in_channels=256,
                                       out_channels=config.hidden_size,
                                       kernel_size=4,
                                       stride=4)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))  [2,768,16,16]
        x1 = x.flatten(2)  # [2,768,256]
        x1 = x1.transpose(-1, -2)  # (B, n_patches, hidden)  [2,256,768]

        embeddings = x1 + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, num_T):
        super(Encoder, self).__init__()
        self.num_T = num_T
        self.vis = vis
        self.embeddings = Embeddings(config)

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)  # InstanceNorm1d(config.hidden_size, eps=1e-6)
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        # self.layer2 = nn.ModuleList()
        # for _ in range(config.transformer["num_layers"]):
        #     layer2 = Block(config, vis)
        #     self.layer2.append(copy.deepcopy(layer2))
        # self.layer3 = nn.ModuleList()
        # for _ in range(config.transformer["num_layers"]):
        #     layer3 = Block(config, vis)
        #     self.layer3.append(copy.deepcopy(layer3))
        # self.layer4 = nn.ModuleList()
        # for _ in range(config.transformer["num_layers"]):
        #     layer4 = Block(config, vis)
        #     self.layer4.append(copy.deepcopy(layer4))
        # self.conv1 = Conv2dReLU(in_channels=1536, out_channels=768, kernel_size=1, stride=1, padding=0)
        # # self.conv1 = torch.nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=1, stride=1, padding=0)
        # self.conv2 = Conv2dReLU(in_channels=2304, out_channels=768, kernel_size=1, stride=1, padding=0)
        # self.conv3 = Conv2dReLU(in_channels=3072, out_channels=768, kernel_size=1, stride=1, padding=0)
        # # self.conv4 = torch.nn.Conv2d(in_channels=3840, out_channels=768, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, emb):
        for layer_block in self.layer:
            hidden_states, _ = layer_block(hidden_states)  # [1,256,768]
        x = self.encoder_norm(hidden_states)  # [1,256,768]
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)  # [1,768,256]
        x1 = x.contiguous().view(B, hidden, h, w)  # [1, 768, 16, 16] = emb
        # x1 = torch.cat([x1, emb], dim=1)
        # x1 = self.conv1(x1)
        # x1 = x1.flatten(2)  # [1,768,256]
        # x1 = x1.transpose(-1, -2)

        # for layer_block in self.layer2:
        #     hidden_states, _ = layer_block(x1)  # [1,256,768]
        # x1 = self.encoder_norm(hidden_states)  # [1,256,768]
        # x1 = x1.permute(0, 2, 1)  # [1,768,256]
        # x1 = x1.contiguous().view(B, hidden, h, w)  # [1, 768, 16, 16]
        # x2 = torch.cat([x1, x, emb], dim=1)
        # x2 = self.conv2(x2)
        # x2 = x2.flatten(2)  # [1,768,256]
        # x2 = x2.transpose(-1, -2)

        # for layer_block in self.layer3:
        #     hidden_states, _ = layer_block(x2)  # [1,256,768]
        # x2 = self.encoder_norm(hidden_states)  # [1,256,768]
        # x2 = x2.permute(0, 2, 1)  # [1,768,256]
        # x2 = x2.contiguous().view(B, hidden, h, w)  # [1, 768, 16, 16]
        # x3 = torch.cat([x2, x1, x, emb], dim=1)
        # x3 = self.conv3(x3)
        # x3 = x3.flatten(2)  # [1,768,256]
        # x3 = x3.transpose(-1, -2)

        # for layer_block in self.layer4:
        #     hidden_states, _ = layer_block(x3)  # [1,256,768]
        # x3 = self.encoder_norm(hidden_states)  # [1,256,768]
        # x3 = x3.permute(0, 2, 1)  # [1,768,256]
        # x4 = x3.contiguous().view(B, hidden, h, w)  # [1, 768, 16, 16]

        return x, x1


class Transformer(nn.Module):
    def __init__(self, config, vis, num_T):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config, vis, num_T)
        # self.encoder2 = Encoder(config, vis, num_T)

    def forward(self, input_ids):
        embedding_output, x = self.embeddings(input_ids)  #[2,64,16,16]
        encoded, x2 = self.encoder(embedding_output, x)  # (B, n_patch, hidden)
        # encoded, x3 = self.encoder2(encoded, x2)
        return x2   # [2,256,768]


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)  #[2,768,256]
        x = x.contiguous().view(B, hidden, h, w)  #[2,768,16,16]
        x = self.conv_more(x)  #[2,512,16,16]
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)  #[2,256,32,32],[2,128,64,64],[2,64,128,128],[2,16,256,256]]
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, num_classes=5, num_T=4, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'seg'
        self.transformer = Transformer(config, vis, num_T)
        self.config = config
        self.conv2d = torch.nn.Conv2d(in_channels=16, out_channels=5, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x3 = self.transformer(x)  # (B, n_patch, hidden)
        return x3


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
}
