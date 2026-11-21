#!/bin/bash -l
# vLLM 0.30.0 for A100 nodes with driver 570 (CUDA 12.8) on RHEL 8 (glibc 2.28).
# - PyPI's default vLLM wheel targets CUDA 13, which this driver cannot run, so
#   use vLLM's +cu129 build and torch's cu129 wheels (CUDA 12.x minor-version
#   compatibility lets 12.9 binaries run on a 12.8 driver; checked on a GPU node).
# - llguidance (structured output only, unused here) has no glibc 2.28 wheel for
#   the pinned version, so vLLM's dependencies are installed without it.
# - The import check runs in the GPU job: importing vLLM on the GPU-less login
#   node can stall for many minutes.
set -euo pipefail
R=${QIL_ROOT:?set QIL_ROOT to the cluster work directory}
module load miniforge3/25.3.1
export PIP_CACHE_DIR=$R/pip-cache
rm -rf $R/venv-vllm
python -m venv $R/venv-vllm
source $R/venv-vllm/bin/activate
pip install -q --upgrade pip packaging
pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
pip install -q --no-deps 'https://wheels.vllm.ai/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
python - > $R/vllm-requirements.txt <<'PY'
from importlib.metadata import requires
from packaging.requirements import Requirement
for line in requires('vllm'):
    req = Requirement(line)
    if req.name.lower() in ('llguidance', 'torch'):
        continue
    if req.marker is None or req.marker.evaluate({'extra': ''}):
        print(str(req).split(';')[0])
PY
echo "--- requirements"; cat $R/vllm-requirements.txt
pip install -q -r $R/vllm-requirements.txt torch==2.13.0 --extra-index-url https://download.pytorch.org/whl/cu129
# The PyPI torchcodec wheel links CUDA 13 (libnvrtc.so.13) and vLLM imports it
# unconditionally, so replace it with PyTorch's cu129 build.
pip install -q --force-reinstall --no-deps torchcodec --index-url https://download.pytorch.org/whl/cu129
# vLLM loads llguidance at engine start even without structured output; 1.6.1 is
# the newest release with a glibc 2.28 wheel (vLLM nominally wants >=1.7).
pip install -q --no-deps --only-binary=:all: llguidance==1.6.1
pip install -q 'pytest>=8,<10'
echo "--- pip check"; pip check || true
echo "--- versions"
pip list 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|triton|transformers|safetensors|vllm|pytest|numpy|xgrammar|flashinfer-python|nvidia-cuda-runtime[-a-z0-9]*) '
python -c "import torch;print('torch build', torch.__version__, 'cuda', torch.version.cuda)"
echo SETUP_DONE
