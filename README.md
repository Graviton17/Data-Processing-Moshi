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
[1] diarize        pyannote 3.1 → speaker segments (audio loaded @ 24 kHz mono)
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

Requires Python ≥ 3.10 and a CUDA-capable GPU for the ML stages.

```sh
cd data-processing
python -m pip install -r requirements.txt        # full stack
# or, editable install with extras:
python -m pip install -e ".[full,dev]"
```

### 2. Get a HuggingFace token (one-time, for pyannote)

pyannote's diarization model is gated. Accept the terms for
[`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
on HuggingFace, create a token, then:

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
| `--manifest-only` | Just (re)build `dataset.jsonl` from an existing `--out` dir. |
| `-v` | Verbose logging (per-file stage timings + drop reasons). |

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
