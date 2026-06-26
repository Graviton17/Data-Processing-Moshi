"""Stage abstraction (Template Method + Chain of Responsibility).

``Stage.__call__`` is the template: it skips already-dropped contexts, logs,
times the work, and delegates the real work to the subclass ``_run``. The
:class:`~dataprep.pipeline.Pipeline` chains stages by feeding each one's output
context into the next.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..models import PipelineContext
from ..utils.logging import get_logger


class Stage(ABC):
    #: Whether the stage should still run on a context that has been dropped.
    runs_on_dropped: bool = False

    def __init__(self) -> None:
        self.log = get_logger(f"stage.{self.name}")

    @property
    def name(self) -> str:
        return type(self).__name__.replace("Stage", "").lower()

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.dropped and not self.runs_on_dropped:
            return ctx
        t0 = time.perf_counter()
        ctx = self._run(ctx)
        dt = time.perf_counter() - t0
        status = f"DROPPED ({ctx.drop_reason})" if ctx.dropped else "ok"
        self.log.debug("%s %s in %.2fs", ctx.stem, status, dt)
        return ctx

    @abstractmethod
    def _run(self, ctx: PipelineContext) -> PipelineContext: ...
