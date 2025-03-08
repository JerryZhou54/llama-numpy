# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from matmul import shiftadd_matmul

def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape, dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape

    # Create lower triangular mask with very low values
    mask = np.full((tgt_len, tgt_len), np.finfo(dtype).min, dtype=dtype)

    # Create the condition mask (equivalent to PyTorch masked_fill_)
    mask_cond = np.arange(mask.shape[-1])
    mask[mask_cond < (mask_cond + 1).reshape(mask.shape[-1], 1)] = 0  # Fill lower triangle with 0

    if past_key_values_length > 0:
        past_mask = np.zeros((tgt_len, past_key_values_length), dtype=dtype)
        mask = np.concatenate([past_mask, mask], axis=-1)  # Append past mask

    # Expand dimensions to match PyTorch behavior
    mask = np.expand_dims(mask, axis=(0, 1))  # (1, 1, tgt_len, tgt_len + past_key_values_length)
    mask = np.tile(mask, (bsz, 1, 1, 1))  # Expand batch dimension

    return mask


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask, dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    # Expand mask to shape (bsz, 1, tgt_len, src_len)
    expanded_mask = np.expand_dims(mask, axis=(1, 2))  # Adds two singleton dimensions
    expanded_mask = np.tile(expanded_mask, (1, 1, tgt_len, 1))  # Broadcast across tgt_len

    # Invert the mask (1.0 - mask)
    inverted_mask = 1.0 - expanded_mask.astype(dtype)

    # Apply masked_fill (replace `True` values with smallest float)
    inverted_mask = np.where(inverted_mask.astype(bool), np.finfo(dtype).min, inverted_mask)

    return inverted_mask

class LlamaConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        nbits=3,
        groupsize=128,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.nbits = nbits
        self.groupsize = groupsize

class NumpyEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        # Initialize embeddings randomly (like nn.Embedding)
        self.weight = np.random.randn(vocab_size, embedding_dim).astype(np.float32)

    def forward(self, indices):
        """Lookup embeddings based on input indices."""
        return self.weight[indices]  # NumPy indexing for lookup

class LlamaRMSNorm():
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        self.weight = np.ones(hidden_size).astype(np.float32)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        
        hidden_states = hidden_states.astype(np.float32)
        variance = np.mean(np.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * (1 / np.sqrt(variance + self.variance_epsilon))
        output = self.weight * hidden_states.astype(input_dtype)
        return output


class LlamaRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequency
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2).astype(np.float32) / self.dim))
        self.inv_freq = inv_freq  # Store as NumPy array

        # Precompute cosine and sine embeddings
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        """Precompute cosine and sine embeddings for given sequence length."""
        self.max_seq_len_cached = seq_len
        t = np.arange(self.max_seq_len_cached, dtype=np.float32)

        # Compute frequencies
        freqs = np.outer(t, self.inv_freq)  # Equivalent to torch.einsum("i,j->ij", t, self.inv_freq)

        # Create cos/sin cache
        emb = np.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = np.cos(emb)[None, None, :, :]  # Add batch dimensions
        self.sin_cached = np.sin(emb)[None, None, :, :]  # Add batch dimensions

    def forward(self, x, seq_len=None):
        """Retrieve cos/sin embeddings for a given sequence length."""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len)

        return (
            self.cos_cached[:, :, :seq_len, :].astype(x.dtype),
            self.sin_cached[:, :, :seq_len, :].astype(x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input (NumPy version)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate([-x2, x1], axis=-1)  # Equivalent to torch.cat

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Applies rotary positional embedding to queries and keys (NumPy version)."""
    
    # The first two dimensions of cos and sin are always 1, so we `squeeze` them.
    cos = np.squeeze(cos, axis=(0, 1))  # [seq_len, dim]
    sin = np.squeeze(sin, axis=(0, 1))  # [seq_len, dim]

    # Select positions from cos/sin cache
    cos = np.expand_dims(cos[position_ids], axis=1)  # [bs, 1, seq_len, dim]
    sin = np.expand_dims(sin[position_ids], axis=1)  # [bs, 1, seq_len, dim]

    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LlamaMLP():
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        K, N = self.hidden_size, self.intermediate_size
        nbits, groupsize = self.config.nbits, self.config.groupsize
        # std = config.initializer_range
        # self.gate_proj = std * np.random.randn(self.hidden_size, self.intermediate_size).astype(np.float32)
        # self.up_proj = std * np.random.randn(self.hidden_size, self.intermediate_size).astype(np.float32)
        # self.down_proj = std * np.random.randn(self.intermediate_size, self.hidden_size).astype(np.float32)
        self.gate_proj_bW = np.random.randint(256, size=(K//8, nbits, N), dtype=np.uint8)
        self.gate_proj_alpha = np.random.randn(K//8, N*8 // groupsize, nbits)
        self.up_proj_bW = np.random.randint(256, size=(K//8, nbits, N), dtype=np.uint8)
        self.up_proj_alpha = np.random.randn(K//8, N*8 // groupsize, nbits)
        self.down_proj_bW = np.random.randint(256, size=(N//8, nbits, K), dtype=np.uint8)
        self.down_proj_alpha = np.random.randn(N//8, K*8 // groupsize, nbits)

    def silu(self, x):
        return x * (1 / (1 + np.exp(-x)))

    def forward(self, x):
        gate_out = shiftadd_matmul(self.gate_proj_bW, self.gate_proj_alpha, x)
        up_out = shiftadd_matmul(self.up_proj_bW, self.up_proj_alpha, x)
        act = self.silu(gate_out) * up_out
        output = shiftadd_matmul(self.down_proj_bW, self.down_proj_alpha, act)

        return output


class LlamaAttention():
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        assert self.num_heads == self.num_key_value_heads
        K, N = self.hidden_size, self.num_heads * self.head_dim
        nbits, groupsize = self.config.nbits, self.config.groupsize
        # std = config.initializer_range
        # self.q_proj = std * np.random.randn(self.hidden_size, self.num_heads * self.head_dim).astype(np.float32)
        # self.k_proj = std * np.random.randn(self.hidden_size, self.num_key_value_heads * self.head_dim).astype(np.float32)
        # self.v_proj = std * np.random.randn(self.hidden_size, self.num_key_value_heads * self.head_dim).astype(np.float32)
        # self.o_proj = std * np.random.randn(self.num_heads * self.head_dim, self.hidden_size).astype(np.float32)
        self.q_proj_bW = np.random.randint(256, size=(K//8, nbits, N), dtype=np.uint8)
        self.k_proj_bW = np.random.randint(256, size=(K//8, nbits, N), dtype=np.uint8)
        self.v_proj_bW = np.random.randint(256, size=(K//8, nbits, N), dtype=np.uint8)
        self.o_proj_bW = np.random.randint(256, size=(N//8, nbits, K), dtype=np.uint8)

        self.q_proj_alpha = np.random.randn(K//8, N*8 // groupsize, nbits)
        self.k_proj_alpha = np.random.randn(K//8, N*8 // groupsize, nbits)
        self.v_proj_alpha = np.random.randn(K//8, N*8 // groupsize, nbits)
        self.o_proj_alpha = np.random.randn(N//8, K*8 // groupsize, nbits)
 
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
    ):
        bsz, q_len, _ = hidden_states.shape

        # query_states = np.dot(hidden_states, self.q_proj)
        # key_states = np.dot(hidden_states, self.k_proj)
        # value_states = np.dot(hidden_states, self.v_proj)
        query_states = shiftadd_matmul(self.q_proj_bW, self.q_proj_alpha, hidden_states)
        key_states = shiftadd_matmul(self.k_proj_bW, self.k_proj_alpha, hidden_states)
        value_states = shiftadd_matmul(self.v_proj_bW, self.v_proj_alpha, hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb.forward(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = np.concatenate([past_key_value[0], key_states], axis=2)
            value_states = np.concatenate([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = np.matmul(query_states, np.swapaxes(key_states, 2, 3)) / math.sqrt(self.head_dim)
    
        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #     )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32

        def softmax(x, axis=-1):
            """ Numerically stable softmax implementation in NumPy """
            x_max = np.max(x, axis=axis, keepdims=True)  # Prevent overflow
            exp_x = np.exp(x - x_max)  # Subtract max for numerical stability
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        attn_weights = softmax(attn_weights, axis=-1).astype(query_states.dtype)
        attn_output = np.matmul(attn_weights, value_states)

        # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        attn_output = np.swapaxes(attn_output, 1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # attn_output = np.dot(attn_output, self.o_proj)
        attn_output = shiftadd_matmul(self.o_proj_bW, self.o_proj_alpha, attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer():
    def __init__(self, config: LlamaConfig):
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ## Added
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = np.random.randn(config.hidden_size, config.vocab_size)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm.forward(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        # hidden_states = hidden_states.cpu().detach().numpy()
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.norm.forward(hidden_states)

        logits = np.dot(hidden_states, self.lm_head)

        outputs = (logits, hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs