# dataprep — raw mono conversations → Moshi two-stream dataset

Turns mono, multi-speaker recordings (mixed quality: some clean, some with
intro/background music) into the **stereo + word-aligned-transcript** format that
[`moshi-finetune`](../moshi-finetune) consumes:

- **Left channel (0)** = agent / main speaker → labelled `SPEAKER_MAIN`
- **Right channel (1)** = user → labelled `SPEAKER_OTHER`
- A sibling `<file>.json` of `{"alignments": [[text, [start, end], speaker], ...]}`
- A `dataset.jsonl` manifest of `{"path", "duration"}`

At training time `moshi-finetune` encodes each channel with Mimi (8 codebooks
each → 16 audio tokens) plus 1 text token = **17 tokens/frame**.

## Pipeline

```
[1] diarize        pyannote 4.x (speaker-diarization-community-1) → speaker
                   segments (audio loaded @ 24 kHz mono)
[2] musicsuppress  Demucs vocals-stem substitution, but only on speech windows
                   whose non-vocal energy ratio exceeds the threshold
[3] clean          DC removal, EBU R128 loudness (-23 LUFS), peak limit.
                   NO pre-emphasis (Mimi is a learned raw-waveform codec)
[4] filter         keep top-2 speakers; drop chunks with >2 speakers or a
                   single-speaker turn >60 s; drop file if too little speech
[5] streamize      mask cleaned mono per speaker → stereo; channel-swap
                   augmentation emits both A=agent and B=agent variants
[6] purity         re-diarize each channel; quarantine variants whose crosstalk
                   exceeds the threshold (catches diarization errors masking can't)
[7] transcribe     WhisperX per channel (once per speaker), label main/user
[8] writeoutputs   stereo WAV + transcript JSON (quarantined → quarantine/)
```

## Input data: what's expected

This pipeline is built for the **mono conversation → finetune** case.

| Requirement | Details |
|---|---|
| **Channels** | **Mono** (single channel). Stereo files are downmixed to mono on load, so don't pre-separate speakers yourself — diarization does that. |
| **Speakers** | Natural **2-person conversations**. Files with a 3rd speaker still work: extra-speaker spans are dropped at the chunk level (top-2 speakers by talk time are kept). |
| **Format** | `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus` (anything `sphn`/`soundfile` can read). |
| **Sample rate** | Any — files are resampled to 24 kHz internally. Higher-quality source is better. |
| **Length** | A few seconds up to hours per file. Long single-speaker stretches (>60 s) are dropped chunk-by-chunk, not the whole file. |
| **Quality** | Mixed is fine. Clean speech is passed through; intros/background music are removed (Demucs) only where detected. |
| **Language** | Set `transcribe.language` in `config.yaml` (default `en`). |
| **Transcripts** | **Not needed** — generated automatically by WhisperX. |

**Good sources:** interviews, podcasts (2 hosts), phone/support calls, scripted
2-person dialogues. **Avoid:** panels/meetings with many overlapping speakers,
music-only tracks, single-speaker monologues/lectures (no turn-taking to learn).

## Adding your data

1. Put all your raw audio anywhere on disk, e.g. `raw_audio/` (subfolders are
   scanned recursively):

   ```
   raw_audio/
   ├── interview_01.mp3
   ├── call_center/
   │   ├── ticket_1234.wav
   │   └── ticket_1235.wav
   └── podcast_ep12.flac
   ```

2. That's it — no manifest or transcripts to prepare. The pipeline discovers the
   files, and the `dataset.jsonl` manifest + per-file `.json` transcripts are
   produced for you.

3. Run the pipeline (next section). It writes the training-ready dataset to
   `dataset/`.

## Design

The code follows a few deliberate patterns so stages and algorithms are easy to
swap and test:

| Pattern | Where |
|---|---|
| **Chain of Responsibility / Pipeline** | `pipeline.Pipeline` runs ordered `Stage`s over a shared `PipelineContext`; stages no-op on a dropped context. |
| **Template Method** | `stages.base.Stage.__call__` handles skip/log/time, delegates to `_run`. |
| **Strategy + Dependency Injection** | `strategies/` defines `Diarizer`, `SourceSeparator`, `MusicDetector`, `Transcriber` ABCs; concrete libs are injected into stages. |
| **Factory** | `factory.build_pipeline(config)` is the only place that picks concrete strategies. |
| **DTO** | `models.PipelineContext` / `StreamVariant` carry state between stages. |

Heavy ML imports (torch, demucs, pyannote, whisperx) are **lazy** — the package
imports and the pure-logic stages run without them, which is what keeps the
filter logic unit-testable.

## How to run

### 1. Install

Requires Python 3.10–3.13 and a CUDA-capable GPU for the ML stages.

The ML stack must be a matched torch-2.8 set (pyannote.audio 4.x hard-pins
torch==2.8.0 / torchaudio==2.8.0 / torchcodec==0.7.0; whisperx needs torch~=2.8).
The torch trio + torchcodec come from a CUDA index, so install them first:

```sh
cd Data-Processing-Moshi
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 torchcodec==0.7.0 \
  --index-url https://download.pytorch.org/whl/cu126   # cu128 for newer drivers
python -m pip install -r requirements.txt        # rest of the stack
# or, editable install with extras:
python -m pip install -e ".[full,dev]"
```

On a fresh Kaggle/Colab box, just run `bash setup_env.sh` (installs the whole
matched set in one pass), then **restart the kernel/runtime**.

### 2. Get a HuggingFace token (one-time, for pyannote)

pyannote 4.x's diarization model is gated. Accept the terms for
[`pyannote/speaker-diarization-community-1`](https://hf.co/pyannote/speaker-diarization-community-1)
**while logged in to HuggingFace** (this is what causes the `GatedRepoError: 401`
if skipped), create a token, then:

```sh
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

(The env var name is configurable via `diarization.hf_token_env`.)

### 3. Run the pipeline

```sh
python -m dataprep.cli --raw raw_audio --config config.yaml -v
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--raw DIR` | Directory of raw audio (scanned recursively). **Required.** |
| `--config FILE` | YAML config (defaults used if omitted). |
| `--out DIR` | Override output directory (default `dataset/`). |
| `--cache DIR` | Override cache directory (default `.cache/`). |
| `--device cuda\|cpu` | Override compute device. |
| `--num-gpus N` | GPUs to use. **Default: auto-detect and use all.** Set `1` to force single process. |
| `--manifest-only` | Just (re)build `dataset.jsonl` from an existing `--out` dir. |
| `-v` | Verbose logging (per-file stage timings + drop reasons). |

#### Multi-GPU (e.g. Kaggle 2x T4)

By default the runner **auto-detects all GPUs and shards the files across them**
(data parallelism): one worker process per GPU, each pinned via
`CUDA_VISIBLE_DEVICES`, each running the full pipeline on its slice of the
files. On 2 GPUs that's ~2x throughput with no extra flags:

```sh
python -m dataprep.cli --raw raw_audio --config config.yaml -v   # uses both T4s
python -m dataprep.cli --raw raw_audio --num-gpus 1              # force single GPU
```

From a Kaggle notebook (Python API):

```python
from dataprep.config import Config
from dataprep.runner import process_dir, detect_num_gpus

print("GPUs:", detect_num_gpus())          # -> 2 on a 2x T4 instance
cfg = Config.from_yaml("config.yaml")
process_dir(cfg, "raw_audio")              # num_gpus=None -> use all
```

Sharding is by whole file, so each GPU loads its own copy of the diarization /
Demucs / WhisperX models. Outputs are keyed by file stem, so shards never
collide, and the `dataset.jsonl` manifest is built once after all workers finish.
*(Note: the heavy models are inference-only here — model-parallel splitting a
single model across GPUs would not help; file-level data parallelism is the
right tool.)*

Re-running is cheap: diarization and transcription are cached per file under
`.cache/`, so only changed stages re-compute.

### 4. Output layout

```
dataset/
├── interview_01__SPEAKER_00_agent.wav     # stereo: L=agent, R=user
├── interview_01__SPEAKER_00_agent.json    # {"alignments": [[text,[s,e],label],...]}
├── interview_01__SPEAKER_01_agent.wav     # channel-swapped variant
├── interview_01__SPEAKER_01_agent.json
├── ...
├── quarantine/                            # variants failing the purity check (WAV only)
├── dataset.jsonl                          # the manifest moshi-finetune reads
└── report.json                            # per-file decisions: drops, music, crosstalk
```

Check `report.json` first to confirm how many files were kept/dropped and why.

### 5. Feed it to moshi-finetune

Point the trainer's data source at the generated manifest:

```yaml
# in moshi-finetune/example/moshi_7B.yaml
data:
  train_data: "/path/to/data-processing/dataset/dataset.jsonl"
  eval_data: ""  # optional
```

```sh
cd ../moshi-finetune
torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml
```

## Test

```sh
python -m pytest -q        # pure filter-logic tests, no GPU/ML deps needed
```

## Tuning notes

- **`music.music_ratio_threshold`** — lower = more aggressive Demucs substitution.
  For mostly-clean corpora you can set `music.enabled: false` to skip Demucs.
- **`purity.max_crosstalk_ratio`** — lower = stricter; inspect `quarantine/` to
  calibrate against your diarization quality.
- **`filters.max_turn_seconds`** / **`max_speakers_per_chunk`** — the two hard
  drop rules.
- Caching: diarization and transcription are cached under `cache_dir` keyed by
  file stem, so re-runs only redo changed stages. (Demucs stems are not cached
  yet — a future optimisation.)
