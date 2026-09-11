"""Shared pytest fixtures.

Restricts anyio-marked async tests to the asyncio backend only - trio isn't
a project dependency, and without this override anyio's default
parametrized ``anyio_backend`` fixture tries both, failing on the missing
trio import.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
