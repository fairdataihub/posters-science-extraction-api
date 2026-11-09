"""GPU scheduling for one shared extraction worker."""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class QueueScheduler:
    """Prefer singles, reserve bulk turns, and rotate environments per class.

    The callback claims and finishes ONE poster. Empty queues do not consume a
    turn. Cursors survive idle polls so a wakeup does not reset fairness.
    """

    environments: list[str]
    single_jobs_per_bulk: int = 3
    singles_processed: int = 0
    _next_environment: dict = field(default_factory=lambda: {"single": 0, "bulk": 0})
    _last_batch: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.single_jobs_per_bulk < 1:
            raise ValueError("SINGLE_JOBS_PER_BULK must be at least 1")
        if len(set(self.environments)) != len(self.environments):
            raise ValueError("Worker environments must be unique")

    def dispatch(self, process: Callable) -> Optional[dict]:
        """Try eligible queues in priority order; return the processed job."""
        classes = ("single", "bulk")
        if self.singles_processed >= self.single_jobs_per_bulk:
            classes = ("bulk", "single")

        for queue_class in classes:
            for offset in range(len(self.environments)):
                index = (self._next_environment[queue_class] + offset) % len(self.environments)
                environment = self.environments[index]
                job = process(environment, queue_class, self._last_batch.get(environment))
                if not job:
                    continue
                self._next_environment[queue_class] = (index + 1) % len(self.environments)
                if queue_class == "bulk":
                    self.singles_processed = 0
                    self._last_batch[environment] = job["batchId"]
                else:
                    self.singles_processed = min(
                        self.singles_processed + 1, self.single_jobs_per_bulk
                    )
                return job
        return None
