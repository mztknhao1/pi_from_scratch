from typing import Protocol, runtime_checkable

from pi_from_scratch.contracts import ObservationBatch, PolicyOutput


@runtime_checkable
class Policy(Protocol):
    """The only policy method a runtime is allowed to call."""

    def predict_chunk(self, observation: ObservationBatch) -> PolicyOutput: ...
