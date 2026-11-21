#!/bin/bash -l
set -euo pipefail
R=${QIL_ROOT:?set QIL_ROOT to the cluster work directory}
module load miniforge3/25.3.1
export PIP_CACHE_DIR=$R/pip-cache HF_HOME=$R/hf
[ -d $R/venv ] || python -m venv $R/venv
source $R/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q safetensors transformers==4.55.2 'pytest>=8,<9' huggingface_hub
pip list 2>/dev/null | grep -iE '^(torch|triton|transformers|safetensors|pytest|nvidia-cuda-runtime)' 
hf download Qwen/Qwen3-0.6B --local-dir $R/models/Qwen3-0.6B >/dev/null
ls -la $R/models/Qwen3-0.6B | head -20
echo SETUP_DONE
