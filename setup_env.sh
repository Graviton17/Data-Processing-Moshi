%%bash
set -euo pipefail

# system deps
apt-get -qq update && apt-get -qq install -y ffmpeg

# Step 1: torch stack FIRST, isolated
pip install -q --no-cache-dir \
  torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Step 2: pyannote + rest WITHOUT re-pinning torch
# --no-deps on torch prevents pyannote from dragging in torch 2.8
pip install -q --no-cache-dir \
  "pyannote.audio>=4.0.5" \
  --no-deps

# Step 3: install pyannote's actual deps manually (minus torch which is already there)
pip install -q --no-cache-dir \
  "asteroid-filterbanks>=0.4" \
  "einops>=0.6" \
  "huggingface_hub>=0.19" \
  "lightning>=2.0" \
  "omegaconf>=2.1" \
  "optuna>=3.0" \
  "pyannote.core>=5.0" \
  "pyannote.database>=5.0" \
  "pyannote.metrics>=3.2" \
  "pyannote.pipeline>=3.0" \
  "rich>=12.0" \
  "semver>=3.0" \
  "soundfile>=0.12" \
  "speechbrain>=1.0"

# Step 4: remaining pipeline deps
pip install -q --no-cache-dir \
  "sphn==0.2.0" \
  "soundfile>=0.14.0" \
  "pyloudnorm>=0.2.0" \
  "demucs-infer>=4.1.2" \
  "whisperx>=3.5.0" \
  "numpy>=2.4,<2.6" \
  "pyyaml>=6.0.3"

# verify
python -c "
import torch, torchaudio, pyannote.audio
print('torch      ', torch.__version__)
print('torchaudio ', torchaudio.__version__)
print('pyannote   ', pyannote.audio.__version__)
print('cuda       ', torch.cuda.is_available())
"