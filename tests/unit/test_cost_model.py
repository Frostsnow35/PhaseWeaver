"""K3: the cost model must prefer keeping cheap contexts and dropping expensive ones."""

from phaseweaver.policy import ClusterSnapshot
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler
from phaseweaver.types import Disposition

CAP = 1000


def snap(bytes_: int, tokens: int, slack: float = 600.0) -> ClusterSnapshot:
    return ClusterSnapshot(
        now=100.0,
        capacity_bytes=CAP,
        resident_bytes=bytes_,
        queued=2,
        reasoning=1,
        slack_by_unit={"u1": slack},
        recent_thrash={},
        resident_bytes_by_unit={"u1": bytes_},
        tokens_by_unit={"u1": tokens},
        prefill_tok_per_s=1000,
        mem_bw_bps=10**9,
    )


def make(**cfg: object) -> SystemScheduler:
    return SystemScheduler(SchedulerConfig(use_cost_model=True, **cfg))  # type: ignore[arg-type]


def test_small_context_with_short_tool_is_retained() -> None:
    # hold = (10/1000)*0.1s = 0.001; recompute = 10/1000 = 0.01 -> keep
    assert make().disposition_for("u1", snap(10, 10), 0.1) is Disposition.RETAIN


def test_large_context_with_long_tool_is_dropped_or_offloaded() -> None:
    # hold = (900/1000)*60 = 54; recompute = 5000/1000 = 5 -> yield
    choice = make().disposition_for("u1", snap(900, 5000), 60.0)
    assert choice in (Disposition.DROP, Disposition.OFFLOAD)


def test_offload_disabled_falls_back_to_drop() -> None:
    choice = make(allow_offload=False).disposition_for("u1", snap(900, 5000), 60.0)
    assert choice is Disposition.DROP


def test_cost_model_is_off_by_default() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.config.use_cost_model is False
