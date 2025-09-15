"""Low-level stdout/stderr filters for noisy native prints (e.g., ENDF warnings).

These utilities install process-wide file-descriptor (fd) filters that capture
native library output written directly to stdout/stderr (fd=1/2), match lines
against patterns/prefixes, and forward only the remaining lines to the original
streams. This works even when output bypasses Python's warnings/logging.

Call ``install_endf_fd_filters()`` as early as possible in your script, ideally
before importing libraries that may emit noisy messages.
"""

from __future__ import annotations

import atexit
import os
import re
import sys
import threading
from typing import Iterable, List, Optional, Pattern, Tuple


def _install_single_fd_filter(
    fd: int,
    patterns: Iterable[str],
    suppress_prefixes: Iterable[str],
) -> None:
    """Install a single fd (1 or 2) line filter in the current process.

    Args:
        fd: File descriptor to filter (1=stdout, 2=stderr).
        patterns: Regex patterns; any match on a line will drop that line.
        suppress_prefixes: If a stripped line starts with any given prefix,
            the line will be dropped.
    """
    if fd not in (1, 2):  # pragma: no cover - defensive programming
        raise ValueError("fd must be 1 (stdout) or 2 (stderr)")

    already = getattr(sys, f"_fd_filter_installed_{fd}", False)
    if already:
        return
    setattr(sys, f"_fd_filter_installed_{fd}", True)

    compiled: List[Pattern[str]] = [re.compile(p) for p in patterns]
    prefixes: Tuple[str, ...] = tuple(suppress_prefixes)

    # Duplicate the original fd and create a pipe
    orig_fd = os.dup(fd)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, fd)  # redirect fd to the pipe write end
    os.close(w_fd)

    # Wrap the original fd with a line-buffered text stream
    orig_stream = os.fdopen(orig_fd, "w", buffering=1, encoding="utf-8", errors="replace")

    def _pump() -> None:
        with os.fdopen(r_fd, "rb", closefd=True) as rf:
            buf = b""
            while True:
                chunk = rf.read(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        s = line.decode("utf-8", "ignore")
                    except Exception:
                        s = repr(line)
                    s_stripped = s.lstrip()
                    drop = s_stripped.startswith(prefixes) or any(r.search(s) for r in compiled)
                    if not drop:
                        orig_stream.write(s + "\n")

    t = threading.Thread(target=_pump, name=f"fd{fd}-filter", daemon=True)
    t.start()

    def _restore() -> None:
        try:
            orig_stream.flush()
        finally:
            try:
                os.dup2(orig_fd, fd)
            finally:
                try:
                    orig_stream.close()
                except Exception:
                    pass

    atexit.register(_restore)


def install_endf_fd_filters(
    filter_stdout: bool = True,
    filter_stderr: bool = True,
    patterns: Optional[Iterable[str]] = None,
    suppress_prefixes: Optional[Iterable[str]] = None,
) -> None:
    """Install ENDF/OpenMC noise filters on stdout/stderr fds for this process.

    Args:
        filter_stdout: Whether to filter fd=1 (stdout).
        filter_stderr: Whether to filter fd=2 (stderr).
        patterns: Regex patterns to drop lines; defaults include LTT(3) warnings
            and GNDS messages.
        suppress_prefixes: Line prefixes to drop (after leading whitespace),
            defaults to ENDF/GNDS file-id prefixes like "n-00".
    """
    default_patterns = [
        r"LTT\s*\(?3\)?\s*for elastic scattering.*Legendre only",
        r"GNDS naming convention",
        r"cross_sections",
    ]
    default_prefixes = ["n-00"]

    pat = list(patterns) if patterns is not None else default_patterns
    pre = list(suppress_prefixes) if suppress_prefixes is not None else default_prefixes

    if filter_stdout:
        _install_single_fd_filter(1, pat, pre)
    if filter_stderr:
        _install_single_fd_filter(2, pat, pre)


