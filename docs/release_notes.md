This release contains the dev-selected SFT, GRPO and no-SFT ablation adapters, their merged BF16 models, and the main model quantized with GPTQ W4A16. The frozen protocol and saved raw answers accompany the code in the repository.

The main SFT → GRPO model scored 63.91% on GSM8K versus the original model's 68.99%; the matched no-SFT GRPO ablation scored 74.37%. The main model scored 36.00% on MATH-500 versus the original model's 55.20%. The negative main result is retained.

vLLM achieved 11.18× the fastest measured HF offline throughput. W4A16 reduced weight-file size 2.70×, with an 8.95 percentage-point GSM8K accuracy cost. Quantized MATH-500 accuracy was 22.20% versus the BF16 main model's 36.00%. Full confidence intervals, exact paired McNemar tests and latency measurements are in `RESULTS.md`.

Download `manifest.json` and the files listed in each artifact's `upload_files`. Archives larger than 1 GiB are split into ordered `.part001`, `.part002`, etc.; concatenate all parts in order, verify the complete archive SHA256 from the manifest, then extract. The individual parts are not separate archives.

See `docs/release_artifacts.md` for extraction and model-loading commands. Each archive includes the upstream Apache-2.0 model license. The main GRPO adapter must be applied to the selected merged SFT base; the merged BF16 and W4A16 archives can be loaded directly.
