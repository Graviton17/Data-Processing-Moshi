#!/usr/bin/env bash
# One-shot environment setup for the dataprep pipeline on a fresh cloud box
# (Kaggle / Colab). Run once per session, then RESTART the kernel/runtime.
#
#   bash setup_env.sh
#
# Why this exists: Kaggle/Colab ship mismatched preinstalled torch/torchaudio/
# torchvision and a stale pyannote.audio. Installing piecemeal causes ABI errors
# (libcudart.so.13, "torchvision has no attribute extension") and a pyannote/
# huggingface_hub version skew (pyannote 3.x cannot talk to huggingface_hub 1.0,
# which dropped use_auth_token). This installs the whole stack as one matched set.
set -euo pipefail

# Adjust to match the box's GPU driver if needed (cu121 / cu118).
CUDA_INDEX="https://download.pytorch.org/whl/cu124"

echo ">> system ffmpeg (for the audio_io decode fallback)"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y ffmpeg >/dev/null 2>&1 || true

echo ">> matched torch / torchaudio / torchvision (same CUDA build)"
pip install -q --no-cache-dir \
  torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url "${CUDA_INDEX}"

echo ">> pipeline deps (torch pins repeated so the resolver can't bump them)"
# pyannote.audio >= 4.0 is REQUIRED on modern Kaggle/Colab: those images now
# preinstall huggingface_hub >= 1.0, which removed the `use_auth_token` kwarg.
# pyannote 3.x calls hf_hub_download(..., use_auth_token=...) and dies on it;
# pyannote 4.x uses the modern `token=` API and reads HF_TOKEN from the env
# automatically. whisperx >= 3.8 also requires pyannote >= 4.0, so they agree.
pip install -q --no-cache-dir \
  torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  "pyannote.audio>=4.0" "demucs>=4.0" "whisperx>=3.8" \
  "sphn==0.1.12" "soundfile>=0.12" "pyloudnorm>=0.1.1" \
  "numpy>=1.24" "pyyaml>=6.0" gdown

echo ">> versions (torch trio must all share the same +cuXXX; pyannote >= 4.0)"
python - <<'PY'
import torch, torchaudio, torchvision, pyannote.audio, huggingface_hub
print("torch       ", torch.__version__)
print("torchaudio  ", torchaudio.__version__)
print("torchvision ", torchvision.__version__)
print("pyannote    ", pyannote.audio.__version__, "(must be >= 4.0)")
print("hf_hub      ", huggingface_hub.__version__)
print("cuda        ", torch.version.cuda, "| available:", torch.cuda.is_available())
PY

echo
echo ">> DONE. Now RESTART the kernel/runtime so Python reimports the new builds,"
echo ">> then set HF_TOKEN and run the pipeline."
