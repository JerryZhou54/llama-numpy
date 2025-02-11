import json

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from modeling_llama import LlamaDecoderLayer, LlamaModel

import torch
from torch.nn import Embedding
import numpy as np

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open("config.json", "r") as f:
        config_data = json.load(f)

    config = LlamaConfig(**config_data)

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    layer = LlamaDecoderLayer(config)

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    # embedding = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    # inputs_embeds = embedding(inputs.input_ids).to("cpu")
    inputs_embeds = model.model.embed_tokens(inputs.input_ids).to("cpu")
    del model
    
    hidden_states = inputs_embeds

    # Attention Mask
    past_key_values_length = 0
    batch_size, seq_length, _ = inputs_embeds.shape
    attention_mask = torch.ones(
        (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
    )
    attention_mask = LlamaModel._prepare_decoder_attention_mask(
        None, attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    # Position IDs
    device = inputs_embeds.device
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # Assume no KV Cache
    past_key_value = None
    output_attentions = True
    use_cache = True

    if config.use_numpy:
        hidden_states = hidden_states.cpu().detach().numpy()
        attention_mask = attention_mask.cpu().detach().numpy()
        position_ids = position_ids.cpu().detach().numpy()
        layer.to("cpu")
    else:
        layer.to(device)

    layer_outputs = layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )

    hidden_states, self_attn_weights, present_key_value = layer_outputs
    print(hidden_states[0, 0, :20])
    print(hidden_states.shape)

    print(self_attn_weights[0, 0, 0, :])
    print(self_attn_weights.shape)

    print(present_key_value[0][0, 0, 0, :20])
    print(present_key_value[0].shape)
    print(present_key_value[1].shape)