import json

from modeling_llama_np import NumpyEmbedding, LlamaDecoderLayer, LlamaConfig, prepare_decoder_attention_mask

import numpy as np

SEQ_LENGTH = 16

def load_weight(layer):
    X1 = layer.input_layernorm.weight.shape[0]
    with open(f"llama2_7b_layer0_data/LayerNorm.txt", "r") as f:
        for i in range(X1):
            layer.input_layernorm.weight[i] = float(f.readline())
            layer.post_attention_layernorm.weight[i] = float(f.readline())
            layer.norm.weight[i] = float(f.readline())

    X1 = layer.self_attn.rotary_emb.inv_freq.shape[0]
    with open(f"llama2_7b_layer0_data/RoPE.txt", "r") as f:
        for i in range(X1):
            layer.self_attn.rotary_emb.inv_freq[i] = float(f.readline())

    X1, X2 = layer.self_attn.k_proj.shape
    with open(f"llama2_7b_layer0_data/Self-Attn.txt", "r") as f:
        for i in range(X1):
            for j in range(X2):
                layer.self_attn.q_proj[j][i] = float(f.readline())
                layer.self_attn.k_proj[j][i] = float(f.readline())
                layer.self_attn.v_proj[j][i] = float(f.readline())
                layer.self_attn.o_proj[j][i] = float(f.readline())

    X1, X2 = layer.mlp.gate_proj.shape
    with open(f"llama2_7b_layer0_data/Mlp.txt", "r") as f:
        for i in range(X2):
            for j in range(X1):
                layer.mlp.gate_proj[j][i] = float(f.readline())
                layer.mlp.up_proj[j][i] = float(f.readline())
                layer.mlp.down_proj[i][j] = float(f.readline())

    X1, X2 = layer.lm_head.shape
    with open(f"llama2_7b_layer0_data/LM_head.txt", "r") as f:
        for i in range(X2):
            for j in range(X1):
                layer.lm_head[j][i] = float(f.readline())

def load_embedding(embedding):
    X1, X2 = embedding.weight.shape
    with open(f"llama2_7b_layer0_data/embedding.txt", "r") as f:
        for i in range(X1):
            for j in range(X2):
                embedding.weight[i][j] = float(f.readline())

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    with open("config.json", "r") as f:
        config_data = json.load(f)

    config = LlamaConfig(**config_data)

    layer = LlamaDecoderLayer(config)

    embedding = NumpyEmbedding(config.vocab_size, config.hidden_size)
    # load_embedding(embedding)
    inputs_embeds = embedding.forward(np.random.randint(config.vocab_size, size=(1, SEQ_LENGTH)))

    # load_weight(layer)

    hidden_states = inputs_embeds
    past_key_values = None

    for i in range(2):
        # Attention Mask
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[1]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        attention_mask = np.ones(
            (batch_size, seq_length_with_past), dtype=bool
        )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # Position IDs
        position_ids = np.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=np.int64
        )
        position_ids = np.expand_dims(position_ids, axis=0).reshape(-1, seq_length)

        output_attentions = True
        use_cache = True

        print("Hidden States: ", hidden_states.shape)
        layer_outputs = layer.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        logits, output_hidden_states, self_attn_weights, present_key_value = layer_outputs
        past_key_values = present_key_value
        print(output_hidden_states[0, -1, :20])
        print(output_hidden_states.shape)

        hidden_states = embedding.forward(np.array([[np.argmax(logits[0, -1, :])]]))