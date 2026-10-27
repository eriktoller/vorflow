"""Shared pytest fixtures for the vorflow test suite."""
import gmsh
import pytest


@pytest.fixture(autouse=True)
def ensure_gmsh_finalized():
    """Ensure gmsh is finalized before and after each test.

    Gmsh keeps global state; a test that fails mid-run would otherwise leak
    an initialized session (with its model contents) into the next test.
    """
    if gmsh.is_initialized():
        gmsh.finalize()
    yield
    if gmsh.is_initialized():
        gmsh.finalize()


@pytest.fixture(autouse=True)
def strict_barriers(monkeypatch):
    """Make the tessellator raise instead of warn when exact barrier faces need its mirror pass or cut.

    The barrier construction should never leave work for that safety net,
    so in tests any such repair is a failure.
    """
    from vorflow import tessellator

    monkeypatch.setattr(tessellator, "STRICT_BARRIERS", True)
