# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Notes

### ReplaySSM in this fork

This fork supports cached speculative state updates for ReplaySSM-capable
hybrid models, including Qwen3.5 and Qwen3.8 Flash Next (`Qwen4Exp`). For example:

```bash
vllm serve /path/to/checkpoint \
    --tensor-parallel-size 4 \
    --mamba-backend triton --mamba-cache-mode align \
    --enable-prefix-caching --use-replayssm-spec --replayssm-buffer-len 32 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
```

`replayssm_buffer_len` controls the replay interval, not the number of draft
tokens. It must be at least `1 + num_speculative_tokens`. The speculative path
requires the Triton Mamba backend and supports `none` or `align` cache modes;
it cannot be combined with `--use-replayssm` or stochastic Mamba-cache rounding.

Native KV offload can be enabled with `--kv-offloading-backend native
--kv-offloading-size 30`. ReplaySSM offload requires Model Runner V2, prefix
caching and `--mamba-cache-mode align`; other KV connectors are rejected.
Only materialized block boundaries are offloaded, because private replay-ring
cursors are not transferred. Routed-expert return uses the upstream auxiliary
output connector; the MTP drafter does not collect target-model routing rows.

### General MTP notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
- When the MTP head shares the target's unquantized lm_head (most MTP models),
  `"draft_lm_head_quantization": "fp8"` or `"nvfp4"` in `--speculative-config`
  gives the drafter a quantized copy of that head (Model Runner V2). The target
  still verifies with its own head, so outputs are unchanged; only the
  acceptance length can move. It helps most on bandwidth-bound GPUs with a
  large vocabulary. The copy costs about one byte (FP8) or 0.56 bytes (NVFP4)
  per lm_head weight, e.g. 0.6 or 0.35 GiB for a 248k x 2560 head; it is
  counted in the model weights, so with `--kv-cache-memory-bytes` leave room
  for it.
