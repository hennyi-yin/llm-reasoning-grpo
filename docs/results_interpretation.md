# 结果解读与简历表述

本次主实验始终指 dev 选出的 SFT checkpoint-436 → GRPO checkpoint-600。
无 SFT 消融选择 checkpoint-400；两条 GRPO 训练都完成 600 步，配置只改变起始模型。
测试结果出来后没有改提示、重选检查点、把消融替换为主模型，或为提高分数重跑测试。
完整置信区间、预先指定的配对检验和性能表见 [RESULTS.md](../RESULTS.md)。

## 后训练结果

主模型 GSM8K 为 63.91%，原始模型为 68.99%，差值 −5.08 pp
（精确 McNemar p = 0.000848553）。MATH-500 为 36.00%，低于原始模型的 55.20%。
因此不能使用需求示例中的 “SFT followed by GRPO improved the base model”。

GRPO 相比 SFT 的 GSM8K 从 59.67% 到 63.91%，增加 4.25 pp
（p = 2.49154e−6，98 道题由错变对，42 道由对变错）。
MATH-500 从 34.20% 到 36.00%，增加 1.80 pp（p = 0.232693）；
这一次运行不能据此断言 GRPO 改善了数学迁移能力。

无 SFT 消融为 GSM8K 74.37%、MATH-500 57.20%，均高于两阶段主模型。
主模型与消融的预先指定配对差值分别为 −10.46 pp 和 −21.20 pp。
这些结果支持在这个配置下重新审视 SFT 起点；不能推广为 “SFT 总是有害”。
消融优于原始模型的描述不附加事后新增的显著性检验。

训练曲线中，按保留的日志分别平均 step 5–100 和 505–600 的 20 个记录，
主实验 correctness reward 为 0.5353 → 0.6194，总奖励为 0.6349 → 0.7192；
消融为 0.7469 → 0.8434 和 0.8387 → 0.9412。
消融的零奖励方差组比例为 0.3925 → 0.5850，说明后期更多采样组不产生相对奖励信号；
截断比例为 0.0416 → 0.0175。这些事后窗口描述只概括训练日志，
不作为模型选择标准，也不替代最终测试。重启前未保存更新区间的重复日志按已有绘图规则排除。

## 格式、长度与正确率

GSM8K boxed 比例从原始模型的 91.36% 提高到 SFT 的 99.55% 和主模型的 99.62%，
但正确率下降。平均输出长度从 303.61 token 变为 102.19 / 102.51 token。
MATH-500 主模型平均 334.82 token，原始模型 598.23 token；截断比例分别为 8.00% 和 4.20%。
这些是伴随现象，不能证明长度缩短导致准确率下降，也不能把格式成功等同于推理成功。

`python -m scripts.analyze_saved_results` 只读取保存的逐题记录，重算预先指定配对的
错→对 / 对→错数量，并按两边是否含 boxed 分组，输出 `results/format_analysis.json`。
它不生成新答案、不改变判分、不用于选择模型；格式分组属于事后描述。
SFT 与主模型在 GSM8K 上分别只有 6 / 5 个未 boxed 输出，
格式缺失数量的变化无法单独解释 56 道题的净正确率增长。
在 SFT → 主 GRPO 的 98 道错→对中，97 道两边均有 boxed；42 道对→错中有 41 道如此。
量化 → BF16 主模型的 222 道错→对中，220 道两边均有 boxed；104 道对→错中有 102 道如此。
这进一步说明 “缺少 boxed” 无法解释全部准确率差值，但仍不能把其余差值作因果分解。
比较无 SFT 消融与原始模型时，还应考虑 boxed 比例变化，不能把全部差值归为潜在推理能力。

README 现在直接列出原始模型 → 无 SFT 消融的完整格式拆分。
这组事后描述性比较单独保存在 `results/format_analysis.json` 的
`posthoc_descriptive_comparisons`，不加入原先十组显著性检验。
原始模型无 boxed、消融有 boxed 的 112 道题中，81 道由错变对；
两边都有 boxed 的 1,202 道题中，81 道由错变对、91 道由对变错，净减少 10 道。
另外两组的净变化为零，因此总变化为 81 − 10 = 71 道（+5.38 pp）。
按全部 1,319 题计，新增 boxed 组贡献 +6.14 pp，共同 boxed 组贡献 −0.76 pp；
共同 boxed 子集自身的准确率为 75.71% → 74.88%。
这些数字支持“测得的净提升集中在新增 boxed 的输出”，
但没有对原始无 boxed 答案重新提取判分，不能证明只改格式即可获得该提升。
也不能把这个结论套用到 SFT → 主 GRPO：那组的净提升全部来自两边均有 boxed 的题。

## 部署收益与量化代价

HF 在 batch 1 / 8 / 16 / 32 的完整扫描中，以 batch 32 的 912.04 输出 token/s 最快。
vLLM 同样使用 512 道训练题、每题固定 256 输出 token，测得 10,196.52 token/s，
因此 E = 11.18×。这是指定硬件和负载下的吞吐比，不是所有负载的加速保证。

W4A16 与 BF16 的在线吞吐分别为：并发 1 时 217.18 / 123.76 token/s，
并发 32 时 3,956.96 / 2,883.76 token/s。
并发 32 的 TTFT p99 为 491.41 / 448.82 ms，量化没有同时改善所有延迟指标。

BF16 / W4A16 权重文件为 3,087,467,144 / 1,142,692,360 字节，G = 2.70×。
vLLM 启动日志中的权重显存为 2.98 / 1.12 GiB；这些数值不代表完整服务显存，
还需容纳 KV 缓存、激活和运行时开销。词嵌入与 lm_head 保持高精度，实际压缩小于 4×。
量化后的 GSM8K 为 54.97%，比主模型下降 F = 8.95 pp（p = 5.70345e−11）。
量化后的 MATH-500 为 22.20%（111 / 500），比 BF16 主模型的 36.00% 下降 13.80 pp。
95% bootstrap 区间为 18.60%–25.80%；与主模型的配对检验 p = 2.91236e−10，
其中 96 道题主模型正确而量化错误，27 道相反。
这是较大的精度代价，应与吞吐和大小收益一起报告，不能把该量化配置描述为精度保持良好。

## 可以使用的简历表述

- Built a reproducible LoRA SFT/GRPO pipeline for Qwen2.5-1.5B-Instruct on one 12 GB GPU, with dev-only checkpoint selection, a matched no-SFT ablation, and 9,095 saved final answers evaluated under a frozen protocol; GRPO improved SFT GSM8K accuracy from 59.67% to 63.91%, while the no-SFT ablation reached 74.37%.
- Raised offline serving throughput 11.18× over the fastest measured HF batch with vLLM; reduced checkpoint weight-file size 2.70× with GPTQ W4A16 at an 8.95 percentage-point GSM8K accuracy cost, measuring TTFT/TPOT p50 and p99 across concurrency settings.

需求占位与主模型保持一致：A = 68.99%，B = 63.91%，C = 55.20%，D = 36.00%，
E = 11.18×，F = 8.95 pp，G = 2.70×。不要把 B / D 改成消融分数。

本次只覆盖单个模型、训练种子和 GPU，使用贪心 pass@1；GSM8K 可能出现在预训练数据中。
配对 p 值按预先指定比较原样报告，没有多重比较校正。没有独立重复训练来估计训练随机性的影响。
