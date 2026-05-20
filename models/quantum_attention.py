# coding=utf-8
"""
Quantum Attention Mechanism - TorchQuantum Implementation

Implements quantum attention using TorchQuantum library.

Current version: Symmetric 2-qubit circuit (SymmetricQAOACircuit)

Circuit structure:
1. RY(pi/4 + enc_scale*q) x RY(pi/4 + enc_scale*k)  - initial encoding
2. RY(gamma_diff*(q-k)) x RY(-gamma_diff*(q-k))       - difference encoding (reversed!)
3. RY(gamma_sum*(q+k)) x RY(gamma_sum*(q+k))          - sum encoding
4. CNOT(0->1) -> RY(alpha*(q+k)) -> CNOT(1->0)        - bidirectional entanglement
5. RX(2*beta) x RX(2*beta)                             - symmetric Mixer
6. Joint measurement: P(|00>) + P(|11>)

Key features:
- Symmetric encoding of query and key
- Bidirectional CNOT entanglement
- Joint measurement preserves fuzzy semantics
- 5 trainable parameters (enc_scale, gamma_diff, gamma_sum, alpha, beta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Softmax
import torchquantum as tq
from math import pi


# ============== Symmetric 2-qubit quantum circuit ==============
class QAOACircuit(tq.QuantumModule):
    """
    Symmetric QAOA circuit - query and key treated equally

    Key features:
    1. Uses (q-k) and (q+k) to encode similarity
    2. Bidirectional CNOT for enhanced interaction
    3. Joint measurement P(|00>) + P(|11>)

    Trainable params: 5 (enc_scale, gamma_diff, gamma_sum, alpha, beta)
    Output: attention score in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # ========== Trainable parameters ==========
        self.enc_scale = nn.Parameter(torch.tensor(0.5))  # encoding scale

        self.gamma_diff = nn.Parameter(torch.randn(1) * 0.1)  # (q-k) encoding
        self.gamma_sum = nn.Parameter(torch.randn(1) * 0.1)   # (q+k) encoding

        self.alpha = nn.Parameter(torch.randn(1) * 0.1)  # entanglement strength
        self.beta = nn.Parameter(torch.randn(1) * 0.1)   # Mixer (shared)

        # ========== Quantum gates ==========
        self.ry_q = tq.RY(has_params=False, trainable=False)
        self.ry_k = tq.RY(has_params=False, trainable=False)
        self.ry_diff_0 = tq.RY(has_params=False, trainable=False)
        self.ry_diff_1 = tq.RY(has_params=False, trainable=False)
        self.ry_sum_0 = tq.RY(has_params=False, trainable=False)
        self.ry_sum_1 = tq.RY(has_params=False, trainable=False)
        self.ry_alpha = tq.RY(has_params=False, trainable=False)
        self.rx_0 = tq.RX(has_params=False, trainable=False)
        self.rx_1 = tq.RX(has_params=False, trainable=False)

        self.cnot_01 = tq.CNOT()  # 0 → 1
        self.cnot_10 = tq.CNOT()  # 1 → 0

    def forward(self, q_device, q_val, k_val):
        """
        Forward pass - symmetric 2-qubit circuit

        Args:
            q_device: TorchQuantum device (2 qubits)
            q_val: query values [batch]
            k_val: key values [batch]

        Returns:
            attention_score: score in [0, 1] [batch]
        """
        batch_size = q_val.size(0)

        # ========== Explicitly reset quantum device to |00> state ==========
        # Prevent state accumulation
        q_device.reset_states(bsz=batch_size)

        # ========== 2. Initial encoding ==========
        self.ry_q(q_device, wires=0, params=pi/4 + self.enc_scale * q_val)
        self.ry_k(q_device, wires=1, params=pi/4 + self.enc_scale * k_val)

        # ========== 2. Difference encoding (q-k) ==========
        # if q ~ k -> diff ~ 0 -> small rotation
        # if q != k -> diff large -> large rotation
        diff = q_val - k_val
        self.ry_diff_0(q_device, wires=0, params=self.gamma_diff * diff)
        self.ry_diff_1(q_device, wires=1, params=-self.gamma_diff * diff)  # reversed!

        # ========== 3. Sum encoding (q+k) ==========
        summ = q_val + k_val
        self.ry_sum_0(q_device, wires=0, params=self.gamma_sum * summ)
        self.ry_sum_1(q_device, wires=1, params=self.gamma_sum * summ)

        # ========== 4. Bidirectional entanglement ==========
        # First entanglement: 0 -> 1
        self.cnot_01(q_device, wires=[0, 1])

        # Intermediate rotation (input-dependent!)
        alpha_params = self.alpha * (q_val + k_val)  # depends on q and k
        self.ry_alpha(q_device, wires=1, params=alpha_params)

        # Second entanglement: 1 -> 0 (reversed!)
        self.cnot_10(q_device, wires=[1, 0])

        # ========== 5. Symmetric Mixer ==========
        # Both qubits use the same parameter
        beta_params = (2 * self.beta).expand(batch_size)
        self.rx_0(q_device, wires=0, params=beta_params)
        self.rx_1(q_device, wires=1, params=beta_params)

        # ========== 6. Joint measurement ==========
        state = q_device.get_states_1d()  # [batch, 4] complex tensor
        # state[:, 0] = <00|psi>, state[:, 1] = <01|psi>
        # state[:, 2] = <10|psi>, state[:, 3] = <11|psi>

        probs = state.abs() ** 2  # [batch, 4]

        # Score = probability of "same-state" outcomes
        # P(|00>) = both tend to |0> -> similar
        # P(|11>) = both tend to |1> -> also similar (reversed but consistent)
        membership = probs[:, 0] + probs[:, 3]  # ∈ [0, 1]

        return membership


# ============== Quantum attention score computation (no aggregation) ==============
def quantum_attention_score_no_aggregation(query_layer, key_layer, q_circuit, agg_dim=16):
    """
    Compute attention score matrix using symmetric QAOA circuit (no aggregation).

    Pipeline:
    1. For each of the first agg_dim dimensions d (0 to agg_dim-1):
       - Extract Q[:, :, :, d] and K[:, :, :, d]
       - Form all query-key pairs
       - Compute S_d via QAOA circuit
    2. No aggregation, keep per-dimension scores

    Args:
        query_layer: [batch, num_heads, seq_len, head_dim]
        key_layer: [batch, num_heads, seq_len, head_dim]
        q_circuit: QAOACircuit instance
        agg_dim: number of dimensions to use (default 16)

    Returns:
        attention_scores: [batch, num_heads, seq_len, seq_len, agg_dim]
        Each dimension d is independent, values in [0,1]
    """
    batch_size, num_heads, seq_len, head_dim = query_layer.shape
    device = query_layer.device

    # Expand to all query-key pairs
    q_expanded = query_layer.unsqueeze(3).expand(-1, -1, -1, seq_len, -1)
    k_expanded = key_layer.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)

    # Compute scores for the first agg_dim dimensions only
    scores_per_dim = []

    for d in range(min(agg_dim, head_dim)):
        # Extract all query-key pairs for dimension d
        q_d = q_expanded[:, :, :, :, d].reshape(-1)
        k_d = k_expanded[:, :, :, :, d].reshape(-1)

        n_pairs = q_d.shape[0]

        # Create quantum device
        q_device = tq.QuantumDevice(
            n_wires=2,  # 2-qubit circuit
            bsz=n_pairs,
            device=device,
            record_op=False  # no op history to save memory
        )

        # Compute score via QAOA circuit (in [0,1])
        score_d = q_circuit(q_device, q_d, k_d)

        # Reshape back to matrix form
        score_d = score_d.reshape(batch_size, num_heads, seq_len, seq_len)
        scores_per_dim.append(score_d)

    # ========== No aggregation, stack all dimensions ==========
    # Output: [batch, num_heads, seq_len, seq_len, agg_dim]
    attention_scores = torch.stack(scores_per_dim, dim=-1)

    return attention_scores


# ============== Quantum attention module (no aggregation) ==============
class QuantumAttention(nn.Module):
    """
    Quantum self-attention mechanism (no aggregation)

    Uses symmetric 2-qubit circuit to compute attention scores.

    Key features:
    - Symmetric 2-qubit circuit
    - 5 trainable params (enc_scale, gamma_diff, gamma_sum, alpha, beta)
    - Symmetric encoding of query and key
    - Bidirectional CNOT entanglement
    - Joint measurement P(|00>) + P(|11>)
    - Output range [0, 1]
    - No aggregation: keeps per-dimension independent scores
    - No scaling: scores naturally in [0,1]
    - Softmax applied after aggregation
    """

    def __init__(self, config, vis):
        super(QuantumAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # ========== Aggregation config ==========
        # Only use the first 16 dimensions for aggregation
        self.agg_dim = 16
        # No scaling factor needed since scores are naturally in [0,1]

        # Q, K, V projection layers
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        # ========== Create symmetric QAOA circuit ==========
        self.q_circuit = QAOACircuit()

    def transpose_for_scores(self, x):
        """Split attention heads"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """
        Forward pass (no aggregation version)

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            attention_output: [batch, seq_len, hidden_size]
            weights: attention weights [batch, num_heads, seq_len_q, seq_len_k, agg_dim]
        """
        # Compute Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        batch_size, num_heads, seq_len, head_dim = query_layer.shape

        # ========== Quantum circuit scores (no aggregation) ==========
        # Output: [batch, num_heads, seq_len, seq_len, agg_dim]
        # Each dimension d has values in [0,1]
        attention_scores = quantum_attention_score_no_aggregation(
            query_layer, key_layer, self.q_circuit, self.agg_dim
        )

        # ========== No scaling ==========
        # Scores naturally in [0,1], no scaling needed

        # ========== Aggregate then Softmax (same as classical ViT) ==========
        # attention_scores: [batch, num_heads, seq_len_q, seq_len_k, agg_dim]
        # Step 1: Aggregate agg_dim (sum)
        attention_scores_agg = attention_scores.sum(dim=-1)  # [batch, num_heads, seq_len_q, seq_len_k]

        # Step 2: Softmax normalization (over seq_len_k, same as classical ViT)
        attention_probs = F.softmax(attention_scores_agg, dim=-1)

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # ========== Compute output ==========
        # attention_probs: [batch, num_heads, seq_len_q, seq_len_k]
        # value_layer: [batch, num_heads, seq_len_k, head_dim]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


# ============== agg_dim variants (fully isolated, original QuantumAttention unchanged) ==============

class QuantumAttentionD32(QuantumAttention):
    """agg_dim=32 variant, uses all dimensions when head_dim=192/6=32"""
    def __init__(self, config, vis):
        super().__init__(config, vis)
        self.agg_dim = 32  # override parent's 16


class QuantumAttentionD8(QuantumAttention):
    """agg_dim=8 variant, uses only the first 8 dimensions"""
    def __init__(self, config, vis):
        super().__init__(config, vis)
        self.agg_dim = 8  # override parent's 16
