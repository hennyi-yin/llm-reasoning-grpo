#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_PATH="${LLM_ENV_PATH:-$HOME/.venvs/llm-reasoning-grpo}"
uv venv "$ENV_PATH" --python python3.12
if [[ -f requirements-lock.txt ]]; then
  uv pip sync --python "$ENV_PATH/bin/python" requirements-lock.txt
else
  uv pip install --python "$ENV_PATH/bin/python" -r requirements.in
  uv pip freeze --python "$ENV_PATH/bin/python" > requirements-lock.txt
fi
QUANT_ENV_PATH="${LLM_QUANT_ENV_PATH:-$HOME/.venvs/llm-reasoning-grpo-quant}"
uv venv "$QUANT_ENV_PATH" --python python3.12
uv pip sync --python "$QUANT_ENV_PATH/bin/python" requirements-quant-lock.txt
source "$ENV_PATH/bin/activate"
source scripts/runtime_env.sh
"$ENV_PATH/bin/python" -m src.environment
"$ENV_PATH/bin/python" -m src.prepare
"$ENV_PATH/bin/python" -m pytest -q
printf 'Environment ready: %s\n' "$ENV_PATH"
