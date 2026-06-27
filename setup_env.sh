#!/usr/bin/env bash
# One-shot environment setup for the dataprep pipeline on a fresh cloud box
# (Kaggle / Colab). Run once per session, then RESTART the kernel/runtime.
#
#   bash setup_env.sh
#
# Why this exists, and what was wrong before:
#   The stack was pinned to torch 2.6, but pyannote.audio 4.x HARD-PINS
#   torch==2.8.0 / torchaudio==2.8.0 / torchcodec==0.7.0, and whisperx requires
#   torch~=2.8.0. So torch 2.6 made the graph unsatisfiable. The old script hid
#   this with `pyannote --no-deps` + a hand-written dep list that OMITTED
#   torchcodec (pyannote 4.x's audio backend) -> import/segfault at runtime.
#
#   Fix: install the whole stack as ONE matched, torch-2.8 set in a single
#   resolver pass, with the torch pins repeated so pip can't bump them.
set -euo pipefail

# CUDA build for the torch wheels. cu126 works on Kaggle 2xT4 (sm_75) and most
# cloud drivers; switch to cu128 if the box has a newer driver/GPU.
CUDA_INDEX="https://download.pytorch.org/whl/cu126"

echo ">> system ffmpeg (required: pyannote 4.x decodes via torchcodec+ffmpeg)"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y ffmpeg >/dev/null 2>&1 || true

echo ">> matched torch trio + torchcodec (one CUDA build, installed first)"
pip install -q --no-cache-dir \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 torchcodec==0.7.0 \
  --index-url "${CUDA_INDEX}"

echo ">> pipeline deps (torch pins repeated so the resolver can't bump them)"
# pyannote.audio 4.x is REQUIRED: modern Kaggle/Colab preinstall huggingface_hub
# >= 1.0, which removed `use_auth_token`. pyannote 3.x dies on it; pyannote 4.x
# uses the modern token= API and reads HF_TOKEN from the env. whisperx 3.8 also
# requires pyannote >= 4.0, so they agree on the same matched set.
# sphn 0.2.0 was never published -> 0.1.12 (the version moshi uses).
pip install -q --no-cache-dir \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 torchcodec==0.7.0 \
  --index-url "${CUDA_INDEX}" \
  --extra-index-url https://pypi.org/simple \
  "pyannote.audio>=4.0.5" \
  "whisperx>=3.8.0" \
  "demucs-infer>=4.1.2" \
  "sphn==0.1.12" \
  "soundfile>=0.14.0" \
  "pyloudnorm>=0.2.0" \
  "numpy>=2.1" \
  "pyyaml>=6.0.3"

echo ">> versions (torch trio + torchcodec must share one +cuXXX; pyannote >= 4.0)"
python - <<'PY'
import torch, torchaudio, torchvision, torchcodec, pyannote.audio, huggingface_hub
print("torch       ", torch.__version__)
print("torchaudio  ", torchaudio.__version__)
print("torchvision ", torchvision.__version__)
print("torchcodec  ", torchcodec.__version__)
print("pyannote    ", pyannote.audio.__version__, "(must be >= 4.0)")
print("hf_hub      ", huggingface_hub.__version__)
print("cuda        ", torch.version.cuda, "| available:", torch.cuda.is_available())
PY

echo
echo ">> DONE. Now:"
echo ">>   1. RESTART the kernel/runtime so Python reimports the new builds."
echo ">>   2. Accept the model licence (logged in to HF):"
echo ">>      https://hf.co/pyannote/speaker-diarization-community-1"
echo ">>   3. export HF_TOKEN=hf_...   then run the pipeline."
