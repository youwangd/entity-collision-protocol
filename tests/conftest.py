"""Top-level pytest conftest.

Registers the Hypothesis profiles referenced by `paper/REPRODUCIBILITY.md §3`:

- ``ci``           — `derandomize=True`, fixed seed; default for headline runs.
- ``dev``          — randomised, fewer examples; for fast local iteration.

The profile is selected via the ``HYPOTHESIS_PROFILE`` env var (defaults to
``ci`` so the paper's "every property test is reproducible" claim holds out
of the box). Override locally with ``HYPOTHESIS_PROFILE=dev pytest ...``.

This module is intentionally minimal — no fixtures, no plugin hooks beyond
the Hypothesis registration. Per-package ``conftest.py`` files inherit from
this one and add their own scoped fixtures.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

# CI / reproducibility profile: deterministic shrinking, fixed examples,
# health-checks tolerant of slow stateful machines.
settings.register_profile(
    "ci",
    derandomize=True,
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# Dev profile: faster, randomised, no health-check suppression so flakes
# surface during local development before they hit CI.
settings.register_profile(
    "dev",
    derandomize=False,
    max_examples=25,
    deadline=None,
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
