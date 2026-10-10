# 本地发布包与模型加载

公开下载入口：[Release v1.0.0](https://github.com/hennyi-yin/llm-reasoning-grpo/releases/tag/v1.0.0)。
下载使用该 Release 附带的 `manifest.json`；实际发布与远端校验记录见 `results/publication.json`。
本地打包清单保留生成时的 `uploaded: false`，发布凭据另存，不覆盖原始清单的哈希。

完整流水线成功后，`scripts.package_release` 在 `release_assets/` 生成下列七个归档和 SHA256 清单。打包不上传文件；实际完成状态、归档哈希和分片名称见 `results/release_package.json` 与 `release_assets/manifest.json`。

| 归档 | 内容 | 建议解压目录 |
|---|---|---|
| `sft-adapter.tar.gz` | dev 选出的 SFT checkpoint-436 适配器 | `checkpoints/sft/checkpoint-436` |
| `grpo-adapter.tar.gz` | dev 选出的主 GRPO checkpoint-600 适配器 | `checkpoints/grpo/checkpoint-600` |
| `ablation-adapter.tar.gz` | dev 选出的无 SFT checkpoint-400 适配器 | `checkpoints/grpo_no_sft/checkpoint-400` |
| `sft-bf16.tar.gz` | 已合并的 SFT 起始模型及 tokenizer | `checkpoints/sft_merged` |
| `final-bf16.tar.gz` | 已合并的 SFT → GRPO 主模型及 tokenizer | `models/final-bf16` |
| `ablation-bf16.tar.gz` | 已合并的无 SFT 消融模型及 tokenizer | `models/ablation-bf16` |
| `final-w4a16.tar.gz` | 从主模型校准得到的 W4A16 模型及 tokenizer | `models/final-w4a16` |

每个归档内部的文件都直接位于根目录，没有额外模型文件夹；先建立目标目录，再用 `tar -xzf ARCHIVE -C DIRECTORY` 解压。保留清单中的源文件和模型许可证，不改写权重、适配器配置或合并来源记录。

大于 1 GiB 的归档另有 `.part001`、`.part002` 等分片。发布时采用清单中该归档的 `upload_files`；下载后按清单顺序拼接，核对完整归档的 `sha256`，再解压。不要把单个分片当成可独立解压的模型。完整 `.tar.gz` 也保留在本地。

## 从公开 Release 下载并校验

以下命令适用于 Linux / WSL 中的新克隆。先安装 GitHub CLI；也可以在上面的 Release 页面手动下载相同文件。只使用主 BF16 模型时，下载清单和三个对应分片即可，不必下载全部模型：

```bash
mkdir -p release_assets
gh release download v1.0.0 --repo hennyi-yin/llm-reasoning-grpo \
  --pattern manifest.json --pattern 'final-bf16.tar.gz.part*' \
  --dir release_assets
```

在项目根目录执行以下标准库脚本。它先逐个核对下载分片，再按清单顺序拼接，并核对完整归档哈希。目标归档已存在时会停止，避免覆盖本地文件。

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('release_assets')
manifest = json.loads((root / 'manifest.json').read_text())
artifact = next(a for a in manifest['artifacts'] if a['name'] == 'final-bf16.tar.gz')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

for part in artifact['upload_files']:
    path = root / part['name']
    if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
        raise RuntimeError(f"Invalid download: {part['name']}")

archive = root / artifact['name']
with archive.open('xb') as out:
    for part in artifact['upload_files']:
        with (root / part['name']).open('rb') as src:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
                out.write(chunk)
if archive.stat().st_size != artifact['bytes'] or digest(archive) != artifact['sha256']:
    raise RuntimeError('Reconstructed archive does not match the manifest')
print('Verified:', archive)
PY
```

清单文件自身的 SHA256 可与 `results/publication.json` 中 `manifest.json` 的发布记录核对。其他 BF16 模型使用各自的归档名称和分片；`final-w4a16.tar.gz` 小于 1 GiB，可直接下载完整归档并按清单校验。然后按下面的目录说明解压和启动服务。

## 直接加载合并模型

用于部署或使用这次实验的实际模型时，优先选择已合并归档；不需要重新应用 LoRA。使用 README 中的锁定环境，在完整流水线结束后再启动额外 GPU 进程。

例如解压本地 BF16 主模型（目标目录应为空）：

```bash
mkdir -p models/final-bf16
tar -xzf release_assets/final-bf16.tar.gz -C models/final-bf16
```

若正在查看已完成实验的原始目录，该模型已经存在，无须再次解压或覆盖。W4A16 的解压方式相同，目标目录是 `models/final-w4a16`。

在主 vLLM 环境中运行本地 W4A16 服务：

```bash
source ~/.venvs/llm-reasoning-grpo/bin/activate
source scripts/runtime_env.sh
vllm serve models/final-w4a16 \
  --host 127.0.0.1 --port 8000 --served-model-name reasoning-w4a16 \
  --dtype bfloat16 --gpu-memory-utilization 0.65 --max-model-len 2048 \
  --seed 42 --no-enable-prefix-caching
```

将模型路径换成 `models/final-bf16` 即可加载主 BF16 模型。此命令是模型使用示例；论文式结果仍来自保存的基准与最终评测记录。不要重新生成已经登记的最终测试答案。

## 使用适配器

SFT 和无 SFT 消融适配器的起始模型是 `Qwen/Qwen2.5-1.5B-Instruct`，revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。主 GRPO 适配器的起始模型是这次 dev 选出的 **已合并 SFT 模型**，即 `sft-bf16.tar.gz`；直接把主 GRPO 适配器应用到原始 Qwen 模型会改变模型。

PEFT 保存的 `adapter_config.json` 保留了原训练机器的 `base_model_name_or_path`。不要依靠它自动查找本地起始模型；显式加载正确的模型后，再传入适配器。无需修改配置文件中的原始路径。

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "checkpoints/sft_merged", dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=False,
)
model = PeftModel.from_pretrained(base, "checkpoints/grpo/checkpoint-600")
```

需要重建合并模型时，可用仓库的 dev 选择记录和同一合并实现，输出到一个新的空目录：

```bash
python -m src.merge --config configs/grpo.yaml --output models/grpo-reconstructed
```

这会检查适配器哈希与 SFT 起始模型来源，并保留原来的实验产物。直接部署已发布的合并模型，可避免加载适配器时的数据类型或合并方式差异。
