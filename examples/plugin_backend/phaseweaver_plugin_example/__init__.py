"""Example third-party backend for PhaseWeaver.

It only uses the public protocol surface plus the mock implementation as a base
class. A real adapter (vLLM, SGLang, ...) would talk to the engine's HTTP API and
report its true capabilities via `BackendCaps`.

Install and use:

    cd examples/plugin_backend
    pip install -e .
    python -c "import phaseweaver; print(phaseweaver.PhaseWeaver(backend='example-device'))"
"""

from __future__ import annotations

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import RealClock


def example_caps() -> BackendCaps:
    """A pretend 4 GiB device; declared honestly so the runtime can degrade."""
    return BackendCaps(
        capacity_bytes=4 * 1024**3,
        block_tokens=16,
        bytes_per_token=57344,
        prefill_tok_per_s=2400,
        decode_tok_per_s=40,
        mem_bw_bps=int(10 * 1024**3),
        supports_offload=False,  # no KV connector on this device
    )


def build_backend() -> MockBackend:
    """Entry point: a zero-argument factory returning an InferenceBackend."""
    return MockBackend(caps=example_caps(), clock=RealClock())
