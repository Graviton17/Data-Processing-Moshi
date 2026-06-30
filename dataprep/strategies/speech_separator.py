"""Target-speaker extraction strategy: isolate each enrolled speaker from a
mono mixture (Strategy pattern, like the diarizer / source-separator / ASR).

Why this exists
---------------
Masking the mono mixture per speaker (the default streamize path) cannot un-mix
two voices that overlap in time -- during overlap each "single-speaker" channel
still contains both voices, which is exactly what the purity stage flags as
crosstalk. For Moshi (a full-duplex model) we must *keep* the overlap but make
each channel clean, which requires real separation, not masking.

Approach B -- enrollment-conditioned extraction
-----------------------------------------------
A blind 2-speaker separator (SepFormer) splits the mixture into stems, but the
stem order is arbitrary and flips between time-chunks. We pin identity by
matching each stem to a per-speaker **enrollment embedding** (ECAPA-TDNN voice
print, computed by the stage from each host's solo speech) via cosine
similarity. So channel A always carries host A, channel B always host B, even
across chunk boundaries -- the "no permutation problem" property of target
speaker extraction, built on a reliable, pip-installable separation backbone.

Install: ``pip install speechbrain`` (pulls a torch-compatible build).
Heavy imports are lazy so the package stays importable without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..models import AudioBuffer


class TargetSpeakerExtractor(ABC):
    @abstractmethod
    def extract(
        self, mixture: AudioBuffer, enrollments: dict[str, AudioBuffer]
    ) -> dict[str, AudioBuffer]:
        """Given a mono ``mixture`` and one enrollment clip per speaker id,
        return one isolated channel per speaker id. Each returned buffer has the
        same sample rate and length as ``mixture``."""


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad a 1-D array to exactly ``n`` samples."""
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def assign_stems(
    stem_embs: list[np.ndarray], spk_embs: dict[str, np.ndarray], speakers: list[str]
) -> dict[int, str]:
    """Greedy max-cosine assignment of separated stems to enrolled speakers.

    Embeddings are assumed L2-normalised, so dot product == cosine similarity.
    Each speaker is matched to at most one stem; returns ``{stem_index: speaker}``.
    """
    pairs = []
    for i, se in enumerate(stem_embs):
        for spk in speakers:
            pairs.append((float(np.dot(se, spk_embs[spk])), i, spk))
    pairs.sort(reverse=True)
    assignment: dict[int, str] = {}
    used_stems: set[int] = set()
    used_spk: set[str] = set()
    for _sim, i, spk in pairs:
        if i in used_stems or spk in used_spk:
            continue
        assignment[i] = spk
        used_stems.add(i)
        used_spk.add(spk)
    return assignment


class SpeechBrainTSExtractor(TargetSpeakerExtractor):
    """SepFormer blind separation + ECAPA-TDNN identity locking.

    Parameters
    ----------
    sep_model / embedding_model
        HuggingFace ids for the SpeechBrain separation and speaker-embedding
        models. The separation model defines ``model_sr`` (8 kHz for the default
        wsj02mix). For Mimi-quality audio prefer a 16 kHz separator and set
        ``model_sr=16000`` -- an 8 kHz model caps channel bandwidth at 4 kHz.
    chunk_seconds
        SepFormer is trained on short utterances; we run it in chunks of this
        length and re-lock identity per chunk via the enrollment embeddings.
    """

    def __init__(
        self,
        sep_model: str = "speechbrain/sepformer-wsj02mix",
        embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        model_sr: int = 8000,
        embedding_sr: int = 16000,
        device: str = "cuda",
        chunk_seconds: float = 10.0,
        cache_dir: str = ".cache/speechbrain",
    ):
        self.sep_model = sep_model
        self.embedding_model = embedding_model
        self.model_sr = model_sr
        self.embedding_sr = embedding_sr
        self.device = device
        self.chunk_seconds = chunk_seconds
        self.cache_dir = cache_dir
        self._sep: Any = None
        self._emb: Any = None

    def _load(self):
        if self._sep is not None:
            return
        try:
            # SpeechBrain >= 1.0 inference namespace.
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                # SpeechBrain 0.5.x fallback.
                from speechbrain.pretrained import (  # type: ignore
                    EncoderClassifier,
                    SepformerSeparation,
                )
            except ImportError as exc:
                raise ImportError(
                    "speech separation requires SpeechBrain. Install it with "
                    "`pip install speechbrain` (a torch-compatible build)."
                ) from exc

        run_opts = {"device": self.device}
        self._sep = SepformerSeparation.from_hparams(
            source=self.sep_model,
            savedir=f"{self.cache_dir}/{self.sep_model.replace('/', '__')}",
            run_opts=run_opts,
        )
        self._emb = EncoderClassifier.from_hparams(
            source=self.embedding_model,
            savedir=f"{self.cache_dir}/{self.embedding_model.replace('/', '__')}",
            run_opts=run_opts,
        )

    def _embed(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """L2-normalised speaker embedding for a mono waveform."""
        import torch

        from ..utils.audio_io import resample

        if sr != self.embedding_sr:
            mono = resample(
                AudioBuffer(samples=mono[None, :], sample_rate=sr), self.embedding_sr
            ).channel(0)
        wav = torch.from_numpy(np.ascontiguousarray(mono))[None, :].to(self.device)
        with torch.no_grad():
            emb = self._emb.encode_batch(wav).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    def _separate_chunk(self, mono_model_sr: np.ndarray) -> list[np.ndarray]:
        """Run SepFormer on one chunk (already at ``model_sr``) -> list of stems."""
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(mono_model_sr))[None, :].to(self.device)
        with torch.no_grad():
            est = self._sep.separate_batch(wav)  # (batch, time, n_src)
        est = est[0].detach().cpu().numpy()      # (time, n_src)
        return [est[:, i].astype(np.float32) for i in range(est.shape[1])]

    def extract(
        self, mixture: AudioBuffer, enrollments: dict[str, AudioBuffer]
    ) -> dict[str, AudioBuffer]:
        from ..utils.audio_io import resample

        self._load()
        speakers = list(enrollments.keys())
        spk_embs = {
            spk: self._embed(buf.channel(0), buf.sample_rate)
            for spk, buf in enrollments.items()
        }

        out_sr = mixture.sample_rate
        n_out = mixture.num_samples
        mix_m = (
            resample(mixture, self.model_sr) if out_sr != self.model_sr else mixture
        ).channel(0)
        total = mix_m.shape[0]
        chunk = max(1, int(self.chunk_seconds * self.model_sr))

        channels_m = {spk: np.zeros(total, dtype=np.float32) for spk in speakers}
        for start in range(0, total, chunk):
            end = min(total, start + chunk)
            stems = self._separate_chunk(mix_m[start:end])
            stem_embs = [self._embed(s, self.model_sr) for s in stems]
            assignment = assign_stems(stem_embs, spk_embs, speakers)
            for i, spk in assignment.items():
                channels_m[spk][start:end] = stems[i]

        out: dict[str, AudioBuffer] = {}
        for spk, ch in channels_m.items():
            buf = AudioBuffer(samples=ch[None, :], sample_rate=self.model_sr)
            if self.model_sr != out_sr:
                buf = resample(buf, out_sr)
            out[spk] = AudioBuffer(samples=_fit_length(buf.channel(0), n_out)[None, :], sample_rate=out_sr)
        return out
