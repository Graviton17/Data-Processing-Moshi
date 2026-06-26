"""dataprep: raw mono conversations -> Moshi two-stream finetuning dataset.

A staged pipeline (Chain of Responsibility) that turns mono multi-speaker
recordings into stereo (agent / user) WAVs + word-aligned transcripts in the
exact format expected by ``moshi-finetune``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
