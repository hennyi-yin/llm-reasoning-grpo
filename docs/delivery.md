# 项目交付与验收索引

项目仓库：[hennyi-yin/llm-reasoning-grpo](https://github.com/hennyi-yin/llm-reasoning-grpo)。
模型下载：[Release v1.0.0](https://github.com/hennyi-yin/llm-reasoning-grpo/releases/tag/v1.0.0)。
持续集成：[CPU protocol tests](https://github.com/hennyi-yin/llm-reasoning-grpo/actions/workflows/cpu-tests.yml)。

全部训练、dev 选择、十组最终评测和部署基准已经完成。最终保存了 9,095 条逐题结果；每个模型/任务组合只生成一次测试答案。报告、代码、配置、依赖锁、原始答案、训练日志和恢复证据均在仓库中，模型权重单独放在 Release。

主 SFT → GRPO 模型未超过原始模型；无 SFT 消融表现最好。项目保留原定主模型和完整负结果，不根据测试结果换模型或补训。准确率、置信区间、McNemar 检验和部署数字以自动生成的 [RESULTS.md](../RESULTS.md) 和 [summary.json](../results/summary.json) 为准。结果解释与可用的真实简历表述见 [results_interpretation.md](results_interpretation.md)。

## 十二项需求的验收证据

| 需求 | 已完成内容 | 仓库证据 |
|---|---|---|
| R1 模型 | 固定 Qwen2.5-1.5B-Instruct revision | [模型配置](../configs/models.yaml)、[环境记录](../results/environment.json) |
| R2 数据划分 | seed 42，6,973 train / 500 dev；测试题不参与训练或量化校准 | [划分 ID](../splits/gsm8k_split.json)、[数据配置](../configs/data.yaml) |
| R3 冻结评估 | 在训练前提交提示、贪心解码、长度、答案提取和判分规则 | [协议](protocol.md)、[冻结清单](frozen_manifest.json)、[评估配置](../configs/eval.yaml) |
| R4 SFT | completion-only LoRA，完成两轮，按 dev 选 checkpoint-436 | [SFT 配置](../configs/sft.yaml)、[选择记录](../results/selected_sft.json) |
| R5 GRPO | 从已合并 SFT 出发，600 步，规则正确性与格式奖励，dev 选 600 | [GRPO 配置](../configs/grpo.yaml)、[选择记录](../results/selected_grpo.json)、[训练曲线](../figures/training_metrics.png) |
| R6 消融 | 从原模型直接 GRPO，其余条件一致；dev 选 400 | [消融配置](../configs/grpo_no_sft.yaml)、[选择记录](../results/selected_ablation.json) |
| R7 最终评估 | 五模型 × GSM8K / MATH-500，各一次；逐题答案和完整账本 | [逐题 CSV](../results/per_problem.csv)、[最终评测目录](../results/final_eval)、[模型登记](../configs/final_models.json) |
| R8 统计 | 10,000 次 bootstrap，95% CI；预定配对比较的精确 McNemar | [完整结果](../RESULTS.md)、[统计实现](../src/stats.py) |
| R9 部署基准 | HF batch 1/8/16/32，vLLM 离线，以及 BF16 并发 1/8/32 | [HF](../results/offline_hf.json)、[vLLM](../results/offline_vllm.json)、[在线](../results/online_bf16.json)、[部署说明](serving.md) |
| R10 量化 | 256 道训练题校准的 GPTQ W4A16，大小、并发 1/32、两任务精度 | [量化记录](../results/quantization.json)、[在线结果](../results/online_w4a16.json)、[Release 加载说明](release_artifacts.md) |
| R11 复现 | 锁定两个 GPU 环境；独立 CPU 环境核对全部保存答案并精确复现报告 | [主锁](../requirements-lock.txt)、[量化锁](../requirements-quant-lock.txt)、[CPU 核验](../results/compatibility/final_cpu_table_reproduction.json)、[CI](../.github/workflows/cpu-tests.yml) |
| R12 显存决定 | 看 dev 前按实际 20 步冒烟测试选定 1.5B / Plan B | [冻结决定](../configs/memory_decision.json)、[执行与恢复记录](execution.md) |

## 在新环境核对已发布结果

下面的核验只需要 CPU，不下载模型、不生成新的测试答案，也不改写已提交的报告。依赖与 GitHub Actions 的 CPU 工作流一致：

```bash
git clone https://github.com/hennyi-yin/llm-reasoning-grpo.git
cd llm-reasoning-grpo
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install pytest==9.1.1 numpy==2.3.5 scipy==1.18.1 PyYAML==6.0.3 math-verify==0.9.0
python -m pytest -q
python -m src.protocol verify
python -m scripts.check_saved_report_cpu --require-minimal-env
```

完整 GPU 实验的环境与命令在 [README](../README.md#reproduce)。如需重新训练，应使用其中的独立实验目录命令，保留本次冻结结果。模型下载、分片校验、解压路径和 vLLM 启动示例在 [release_artifacts.md](release_artifacts.md)。

## 发布记录

公开发布与下载核验已完成：七套模型以 14 个文件上传，总计 8,544,311,857 字节。
所有远端文件的大小、状态和 SHA256 与本地清单一致；不带认证凭据的清单下载也通过校验。
验证记录见 [public_release_download.json](../results/compatibility/public_release_download.json)。

`results/release_package.json` 记录七套模型归档的大小和 SHA256；原始清单的 `uploaded: false` 表示本地打包时状态。
`results/publication.json` 单独记录实际公开发布状态、14 个上传文件、GitHub 返回的 SHA256 和下载地址。公开交付完成以该记录的 `status: published` 及 Release 可见性为准。
v1.0.0 标签固定实验发布版本；随后提交到 main 的发布回执记录远端上传结果，不移动已有标签。
