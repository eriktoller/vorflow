"""Package-wide logging setup.

vorflow reports progress through the standard :mod:`logging` module under the
``"vorflow"`` logger. Messages keep the plain look of the old ``print()``
output, but can now be silenced, made more verbose, or redirected to a file
without touching library code.

Verbosity mapping (same scale MeshGenerator has always documented):

- ``0`` — silent: only warnings and errors are shown.
- ``1`` — basic progress messages (the package default).
- ``2`` — debug diagnostics (the ``[DIAG]`` output).

By default vorflow prints to the console through its own handler and does not
propagate to the root logger, so applications that configure logging do not
see messages twice. Call ``set_verbosity(level, console=False)`` to drop the
console handler and let messages propagate to the application's handlers
(this is also what pytest's ``caplog`` needs).
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager

LOGGER_NAME = "vorflow"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

_LEVELS = {0: logging.WARNING, 1: logging.INFO}

_handler = None


def _level_for(verbosity: int) -> int:
    """Logging level for a verbosity value (2 or more means DEBUG)."""
    return _LEVELS.get(int(verbosity), logging.DEBUG)


def set_verbosity(verbosity, console=True):
    """Set how talkative vorflow is.

    Args:
        verbosity (int): 0 = warnings/errors only, 1 = progress messages
            (default), 2 or more = debug diagnostics.
        console (bool): If True (default), vorflow prints through its own
            console handler and does not propagate to the root logger. If
            False, the console handler is removed and messages propagate to
            the application's logging configuration instead.
    """
    global _handler
    logger = logging.getLogger(LOGGER_NAME)
    if console and _handler is None:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_handler)
        logger.propagate = False
    elif not console and _handler is not None:
        logger.removeHandler(_handler)
        _handler = None
        logger.propagate = True
    logger.setLevel(_level_for(verbosity))


def current_verbosity() -> int:
    """Verbosity implied by the vorflow logger's effective level."""
    level = logging.getLogger(LOGGER_NAME).getEffectiveLevel()
    if level <= logging.DEBUG:
        return 2
    if level <= logging.INFO:
        return 1
    return 0


@contextmanager
def verbosity_scope(verbosity):
    """Temporarily set the vorflow log level; None leaves it unchanged."""
    if verbosity is None:
        yield
        return
    logger = logging.getLogger(LOGGER_NAME)
    previous = logger.level
    logger.setLevel(_level_for(verbosity))
    try:
        yield
    finally:
        logger.setLevel(previous)


def warn_user(message: str, category: type = UserWarning, package_dir: str = PACKAGE_DIR) -> None:
    """``warnings.warn`` attributed to the first caller outside vorflow.

    A fixed ``stacklevel`` is wrong whenever the same warning can be reached
    along call chains of different depth (``ConceptualMesh.generate()``
    directly, or through ``MeshGenerator.generate()``), so count the vorflow
    frames instead.
    """
    level, frame = 2, sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(package_dir):
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)
