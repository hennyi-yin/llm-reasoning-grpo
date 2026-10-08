# Serving and quantization evidence

Run these stages after the final GRPO adapter is selected on dev and merged to `models/final-bf16`.
Run each engine in a separate process so its GPU allocations are released before the next engine.

```bash
python -m src.bench_hf --config configs/bench.yaml
python -m src.bench_vllm --config configs/bench.yaml
bash scripts/serve_bench.sh --precision bf16
# In the isolated, locked quantization environment:
python -m src.quantize --config configs/quantize.yaml
# Back in the vLLM environment:
bash scripts/serve_bench.sh --precision w4a16
python -m src.report --update-readme
```

Both offline engines receive the same first 512 training IDs and exactly the same tokenized chat
prompts. Each batch configuration completes one full warmup and three timed full-workload passes.
Every output must contain exactly 256 tokens. HF tests batches 1, 8, 16, 32; only actual CUDA OOMs
permit a batch to be excluded. E is computed against the fastest successful batch. Tokenization is
outside both timers, while generation/request submission and transfer of prepared inputs are timed.
vLLM prefix caching is disabled so repeated passes still execute prefill.

The online wrapper runs the official `vllm bench serve` command. It validates all 500 requests,
the actual 256-token input and output lengths, and finite TTFT/TPOT percentiles. It saves the raw
detailed JSON, startup logs, CLI arguments and hashes. BF16 uses concurrency 1/8/32; W4A16 uses 1/32.
No online measurement is used as a substitute for offline E.

Version-specific choices were verified against [vLLM 0.31.0 serving CLI source](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/serve.py)
and [its random dataset implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/datasets/datasets.py):
temperature is explicitly zero, random range ratio is zero for fixed lengths, and detailed saving
retains the actual lengths. Loaded-model memory is parsed from the [GPU runner's startup log](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_model_runner.py).
Missing weight-memory logs remain null; process memory and KV-cache memory are never substituted.

GPTQ uses W4A16 on Linear layers while excluding `lm_head`, and exactly 256 training questions
with the same chat template. The import and saving APIs follow the [llm-compressor 0.14.0 example](https://github.com/vllm-project/llm-compressor/blob/0.14.0/examples/quantization_w4a16/llama3_example.py).
G is the BF16 / compressed checkpoint size ratio using only standard referenced HF weight files,
excluding tokenizer, optimizer and other metadata. It is a file-size ratio; vLLM's loaded weight
memory is reported separately. F is the paired GSM8K accuracy difference, with exact McNemar p.

## Cross-environment serialization check

The quantization lock uses compressed-tensors 0.19.0; vLLM 0.31.0 uses 0.17.0. A CPU-only
compatibility check on 2026-10-08 generated a random, one-layer Qwen2 model (hidden size 128,
vocabulary 64) using the exact W4A16 GPTQ recipe and synthetic tokens. Its seven Linear modules
were serialized by llm-compressor 0.14.0 / compressed-tensors 0.19.0, with `lm_head` unquantized.
The serving environment's 0.17.0 QuantizationConfig and QuantizationScheme accepted the saved
config, and vLLM 0.31.0's CompressedTensorsConfig parsed it successfully. CPU dequantization of
all seven modules in 0.17.0 matched the 0.19.0 reference tensors bit for bit. CUDA was hidden
for both processes; no real checkpoint, training data, benchmark, or test evaluation was used.

This verifies the serialized schema and standard symmetric INT4 packing used here. A subsequent
synthetic GPU check generated four tokens on the RTX 5070 with vLLM0.31.0 / torch2.13.0;
`results/compatibility/quantization_gpu.json` records the successful kernel execution. It does not establish
compatibility for other quantization formats. The saved scheme is group-size 128, static
activation ordering, with `weight_packed` (int32), `weight_scale` (BF16) and `weight_shape`.
The [0.17.0 packed compressor](https://github.com/vllm-project/compressed-tensors/blob/0.17.0/src/compressed_tensors/compressors/pack_quantized/base.py),
[0.19.0 packed compressor](https://github.com/vllm-project/compressed-tensors/blob/0.19.0/src/compressed_tensors/compressors/pack_quantized/base.py),
and [vLLM's WNA16 weight loader](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)
support this representation; [vLLM's config parser](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)
accepts static activation ordering and rejects unsupported group/dynamic ordering.

The lightweight evidence is saved in `results/compatibility/quantization_cpu.json`; it is
separate from `results/quantization.json`, which is reserved for the real R10 checkpoint.
To repeat the compatibility check without accessing CUDA:

```bash
~/.venvs/llm-reasoning-grpo-quant/bin/python scripts/check_quant_compat.py produce
# Copy the temporary CPU_COMPAT_CHECKPOINT path printed above:
~/.venvs/llm-reasoning-grpo/bin/python scripts/check_quant_compat.py consume \
  --checkpoint /tmp/llm-w4a16-schema-cpu-PRINTED_ID
# Run the GPU check when no other task is using the GPU:
~/.venvs/llm-reasoning-grpo/bin/python -m scripts.check_quant_gpu
```

`src.report` refuses differing model artifacts or frozen protocols across accuracy, quantization
and serving, verifies raw-output hashes and recomputes CSV correctness, boxed status, token count,
truncation and extraction from the saved raw generations, and checks calibration IDs against the train split.
The report does not reread test datasets or generate answers. Incomplete task results are labeled
with their sample count and do not populate resume values. Missing experiments produce null values
and no synthetic curves. Figures are created only when real training/dev/serving records exist.
