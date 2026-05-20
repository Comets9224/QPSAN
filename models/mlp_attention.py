# coding=utf-8
"""
MLP Attention Mechanism - Classical Ablation Study (~50 params)

Uses a small MLP for nonlinear attention scoring (classical ablation of quantum attention).

Comparison:
- Quantum attention: 2-qubit circuit computes attention score, 5 params
- MLP attention: small MLP computes attention score, ~50 params
- Gaussian attention: Gaussian function computes attention score, 1 param

Shared properties:
- Attention score in [0,1]
- Aggregates 16 dimensions
- Sum + Softmax normalization

MLP design:
- Input: [q, k, q-k, q+k] (4 features, matching quantum circuit's 3-layer encoding)
- Structure: Linear(4,8) -> Tanh -> Linear(8,1) -> Sigmoid
- Params: 4x8+8 + 8x1+1 = 49
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout


# ============== MLP scoring function ==============
class MLPScoreFunction(nn.Module):
    """
    MLP scoring function - 49-param ablation

    Compared to quantum circuit:
    - Quantum: 5 params, (q,k) -> 3-layer encoding + entanglement + measurement -> [0,1]
    - MLP: 49 params, (q,k) -> feature construction + 2-layer MLP -> [0,1]

    Input features: [q, k, q-k, q+k] (4 features)
    - q, k: correspond to quantum circuit's initial encoding (enc_scale * q/k)
    - q-k: corresponds to difference encoding (gamma_diff * (q-k))
    - q+k: corresponds to sum encoding (gamma_sum * (q+k))
    Structure: Linear(4,8) -> Tanh -> Linear(8,1) -> Sigmoid
    Params: 4x8+8 = 40 (fc1) + 8x1+1 = 9 (fc2) -> total 49
    """

    def __init__(self):
        super().__init__()
        # Linear(4, 8): 4×8 weights + 8 biases = 40 parameters
        self.fc1 = nn.Linear(4, 8)
        # Linear(8, 1): 8×1 weights + 1 bias = 9 parameters
        self.fc2 = nn.Linear(8, 1)
        # Total params: 49 ~ 50

    def forward(self, q, k):
        """
        Forward pass

        Args:
            q: query value, shape: [...]
            k: key value, shape: [...]

        Returns:
            score: attention score in [0,1], shape: [...]
        """
        # Feature construction (matching quantum circuit's 3-layer encoding)
        x = torch.stack([q, k, q - k, q + k], dim=-1)
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


# ============== MLP attention score computation ==============
def mlp_attention_score_no_aggregation(query_layer, key_layer, mlp_scorer, agg_dim=16):
    """
    Compute attention score matrix using MLP (no aggregation).

    Same interface as quantum_attention_score_no_aggregation.

    Args:
        query_layer: [batch, num_heads, seq_len, head_dim]
        key_layer: [batch, num_heads, seq_len, head_dim]
        mlp_scorer: MLPScoreFunction instance
        agg_dim: number of dimensions to use (default 16)

    Returns:
        attention_scores: [batch, num_heads, seq_len, seq_len, agg_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query_layer.shape

    # Expand to all query-key pairs
    q_expanded = query_layer.unsqueeze(3).expand(-1, -1, -1, seq_len, -1)
    k_expanded = key_layer.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)

    # Compute scores for the first agg_dim dimensions only
    scores_per_dim = []

    for d in range(min(agg_dim, head_dim)):
        q_d = q_expanded[:, :, :, :, d]  # [batch, num_heads, seq_len_q, seq_len_k]
        k_d = k_expanded[:, :, :, :, d]

        # MLP scoring: (q_d, k_d) -> score_d in [0,1]
        score_d = mlp_scorer(q_d, k_d)

        scores_per_dim.append(score_d)

    # Stack all dimensions
    attention_scores = torch.stack(scores_per_dim, dim=-1)
    return attention_scores


# ============== MLP attention module ==============
class MLPAttention(nn.Module):
    """
    MLP self-attention mechanism (classical ablation of quantum attention)

    Uses ~50-param MLP instead of quantum circuit for attention scoring.

    Key features:
    - 49 trainable params (MLP scoring function)
    - Output range [0,1]
    - Aggregates 16 dimensions
    - Sum + Softmax normalization
    """

    def __init__(self, config, vis):
        super(MLPAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Aggregation config (same as quantum attention)
        self.agg_dim = 16

        # Q, K, V projection layers (same as quantum/classical)
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        # MLP scoring function (49 params)
        self.mlp_scorer = MLPScoreFunction()

    def transpose_for_scores(self, x):
        """Split attention heads"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            attention_output: [batch, seq_len, hidden_size]
            weights: attention weights
        """
        # Compute Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # MLP attention scores
        attention_scores = mlp_attention_score_no_aggregation(
            query_layer, key_layer, self.mlp_scorer, self.agg_dim
        )

        # Aggregate + Softmax (same as quantum attention)
        attention_scores_agg = attention_scores.sum(dim=-1)
        attention_probs = F.softmax(attention_scores_agg, dim=-1)

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # Compute output
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


# ============== MLP585 scoring function ==============
class MLPScoreFunction585(nn.Module):
    """
    MLP scoring function - 585-param ablation

    Compared to MLPScoreFunction (49 params):
    - 49 params: Linear(4,8) -> Tanh -> Linear(8,1) -> Sigmoid
    - 585 params: Linear(4,64) -> Tanh -> Linear(64,4) -> Tanh -> Linear(4,1) -> Sigmoid

    Param count:
    - fc1: 4x64 + 64 = 320
    - fc2: 64x4 + 4 = 260
    - fc3: 4x1 + 1 = 5
    - Total: 585
    """

    def __init__(self):
        super().__init__()
        # fc1: 4×64 + 64 = 320 parameters
        self.fc1 = nn.Linear(4, 64)
        # fc2: 64×4 + 4 = 260 parameters
        self.fc2 = nn.Linear(64, 4)
        # fc3: 4×1 + 1 = 5 parameters
        self.fc3 = nn.Linear(4, 1)
        # Total params: 320 + 260 + 5 = 585

    def forward(self, q, k):
        """
        Forward pass

        Args:
            q: query value, shape: [...]
            k: key value, shape: [...]

        Returns:
            score: attention score in [0,1], shape: [...]
        """
        # Feature construction [q, k, q-k, q+k] (matching quantum circuit's 3-layer encoding)
        x = torch.stack([q, k, q - k, q + k], dim=-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(-1)


class MLPAttention585(nn.Module):
    """
    MLP self-attention - 585-param version

    Same as MLPAttention, only replaces scorer with MLPScoreFunction585.
    """

    def __init__(self, config, vis):
        super(MLPAttention585, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Aggregation config (same as 49-param version)
        self.agg_dim = 16

        # Q, K, V projection layers
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        # 585-param scoring function
        self.mlp_scorer = MLPScoreFunction585()

    def transpose_for_scores(self, x):
        """Split attention heads"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            attention_output: [batch, seq_len, hidden_size]
            weights: attention weights
        """
        # Compute Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # MLP585 attention scores
        attention_scores = mlp_attention_score_no_aggregation(
            query_layer, key_layer, self.mlp_scorer, self.agg_dim
        )

        # Aggregate + Softmax
        attention_scores_agg = attention_scores.sum(dim=-1)
        attention_probs = F.softmax(attention_scores_agg, dim=-1)

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # Compute output
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights
