# coding=utf-8
"""
MLP Vision Transformer

Uses a small MLP for nonlinear attention scoring (classical ablation of quantum attention).

Differences from Quantum ViT:
- Quantum ViT: QuantumAttention (5 params, quantum circuit)
- MLP ViT: MLPAttention (~50 params, classical MLP)
- All other parts identical (Embeddings, MLP, LayerNorm, etc.)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

import models.configs as configs

from .ResNet import ResNetV2
from .mlp_attention import MLPAttention, MLPAttention585
from .VIT import Mlp, Embeddings, ACT2FN, np2th

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


class MLPBlock(nn.Module):
    """
    Transformer Block with MLP attention

    Differs from QuantumBlock:
    - Uses MLPAttention instead of QuantumAttention
    - MLP, LayerNorm, residual connections unchanged
    """

    def __init__(self, config, vis):
        super(MLPBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        # ========== Key change: MLPAttention replaces QuantumAttention ==========
        self.attn = MLPAttention(config, vis)

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
        """Load pretrained weights (MLP only; MLP attention trains from scratch)"""
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # attention weights skipped (MLP attention has different structure)
            # only load MLP weights
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


class MLPEncoder(nn.Module):
    """Encoder with MLP attention"""

    def __init__(self, config, vis):
        super(MLPEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = MLPBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class MLPTransformer(nn.Module):
    """Transformer with MLP attention"""

    def __init__(self, config, img_size, vis, in_channels=3):
        super(MLPTransformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = MLPEncoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class MLPVisionTransformer(nn.Module):
    """
    Full MLP Vision Transformer

    Uses a small MLP for nonlinear attention scoring (ablation of quantum attention).

    Architecture:
    - Embeddings: same as quantum ViT (patch + position embedding)
    - MLPEncoder: MLPBlock with MLPAttention (49 params)
    - Classification head: same as quantum ViT
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, in_channels=3):
        super(MLPVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = MLPTransformer(config, img_size, vis, in_channels=in_channels)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        """Load pretrained weights (Embeddings + MLP only; MLP attention trains from scratch)"""
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == 'token':
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                from scipy import ndimage
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


# MLP ViT configs (identical architecture to quantum configs for fair comparison)
MLP_CONFIGS = {
    'ViT-MLP-FER2013': configs.get_fer2013_config(),  # same as ViT-Quantum-FER2013
    'ViT-MLP585-FER2013': configs.get_fer2013_config(),  # same architecture, 585-param scorer
}


# ============== 585-param MLP ViT model ==============

class MLPBlock585(MLPBlock):
    """
    Transformer Block with 585-param MLP attention

    Inherits MLPBlock, only replaces attention layer with MLPAttention585.
    """
    def __init__(self, config, vis):
        super().__init__(config, vis)
        # replace attention layer: 49-param -> 585-param
        self.attn = MLPAttention585(config, vis)


class MLPEncoder585(nn.Module):
    """Encoder with 585-param MLP attention"""

    def __init__(self, config, vis):
        super(MLPEncoder585, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = MLPBlock585(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class MLPTransformer585(nn.Module):
    """Transformer with 585-param MLP attention"""

    def __init__(self, config, img_size, vis, in_channels=3):
        super(MLPTransformer585, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = MLPEncoder585(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class MLPVisionTransformer585(MLPVisionTransformer):
    """
    585-param MLP Vision Transformer

    Inherits MLPVisionTransformer, only replaces transformer with 585-param version.
    """
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, in_channels=3):
        super().__init__(config, img_size, num_classes, zero_head, vis, in_channels)
        # replace transformer with 585-param version
        self.transformer = MLPTransformer585(config, img_size, vis, in_channels=in_channels)
