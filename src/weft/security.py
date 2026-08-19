"""Security-sensitive filesystem and terminal helpers for Weft."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
_TERMINAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class WeftSecurityError(ValueError):
    """Raised when an operation would cross a configured security boundary."""


class UnsafeVaultPathError(WeftSecurityError):
    """Raised for a vault entry that escapes the vault or uses a symlink."""


class UnsafeWriteError(WeftSecurityError):
    """Raised when a sensitive write target is unsafe."""


def vault_root(path: Path) -> Path:
    """Return a canonical existing vault directory."""
    try:
        root = Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise UnsafeVaultPathError(f"Vault does not exist: {path}") from exc
    if not root.is_dir():
        raise UnsafeVaultPathError(f"Vault is not a directory: {path}")
    return root


def _reject_symlink_components(path: Path, root: Path) -> None:
    """Reject symlinks below ``root`` so reads cannot leave the vault."""
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise UnsafeVaultPathError(f"Path is outside the vault: {path}") from exc

    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise UnsafeVaultPathError(f"Vault symlinks are not allowed: {relative}")


def read_vault_text(path: Path, root: Path) -> str:
    """Read one regular UTF-8 file without following vault symlinks."""
    path = Path(path)
    root = Path(root).resolve(strict=True)
    _reject_symlink_components(path, root)

    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise UnsafeVaultPathError(f"Vault entry escapes the vault: {path}") from exc

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise UnsafeVaultPathError(f"Could not safely open vault entry: {path}") from exc

    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UnsafeVaultPathError(f"Vault entry is not a regular file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _prepare_parent(path: Path) -> Path:
    parent = path.parent
    if parent.is_symlink():
        raise UnsafeWriteError(f"Refusing to write through symlink directory: {parent}")
    if parent.exists() and not parent.is_dir():
        raise UnsafeWriteError(f"Write parent is not a directory: {parent}")
    existed = parent.exists()
    parent.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    if not existed or parent.name == ".weft":
        parent.chmod(PRIVATE_DIR_MODE)
    return parent


def _validate_replace_target(path: Path) -> None:
    if path.is_symlink():
        raise UnsafeWriteError(f"Refusing to replace symlink: {path}")
    if path.exists() and not path.is_file():
        raise UnsafeWriteError(f"Refusing to replace non-regular file: {path}")


def secure_write_binary(
    path: Path,
    writer: Callable[[BinaryIO], None],
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically write a mode-0600 file without following the destination."""
    path = Path(path)
    parent = _prepare_parent(path)
    if overwrite:
        _validate_replace_target(path)
    elif path.exists() or path.is_symlink():
        raise FileExistsError(path)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                raise FileExistsError(path) from None
            temporary.unlink()
        path.chmod(PRIVATE_FILE_MODE)
        return path
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def secure_write_text(path: Path, text: str, *, overwrite: bool = True) -> Path:
    """Write UTF-8 text with the same guarantees as :func:`secure_write_binary`."""

    def write(handle: BinaryIO) -> None:
        handle.write(text.encode("utf-8"))

    return secure_write_binary(path, write, overwrite=overwrite)


def secure_append_json(path: Path, record: dict) -> None:
    """Append one JSON object to a mode-0600 JSONL file without following links."""
    path = Path(path)
    _prepare_parent(path)
    if path.is_symlink():
        raise UnsafeWriteError(f"Refusing to append through symlink: {path}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UnsafeWriteError(f"Refusing to append to non-regular file: {path}")
        os.fchmod(fd, PRIVATE_FILE_MODE)
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written == 0:
                raise OSError(f"Could not append complete audit record to {path}")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def terminal_safe(text: str) -> str:
    """Remove terminal control characters while preserving newlines and tabs."""
    return _TERMINAL_CONTROL_RE.sub("", str(text))
