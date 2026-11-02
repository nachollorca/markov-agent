"""Persistent settings for markov: the model and the provider API keys in ``~/.markov/auth.json``.

Schema::

    {"model": "mistral:mistral-large-4", "keys": {"MISTRAL_API_KEY": "..."}}

The file is written atomically (temp file in the same directory + ``os.replace``) with mode ``0600``
inside a ``0700`` directory. Stored keys are merged into ``os.environ`` at startup, but a variable
already set in the real environment wins, so ``MISTRAL_API_KEY=... markov`` keeps working.
Key values are never logged or printed anywhere.
"""

import contextlib
import json
import os
import stat
import tempfile
import warnings
from pathlib import Path
from typing import Any

AUTH_DIR = Path.home() / ".markov"
AUTH_FILE = AUTH_DIR / "auth.json"


def load(path: Path | None = None) -> dict[str, Any]:
    """Read the settings file.

    Args:
        path: File to read; defaults to :data:`AUTH_FILE`.

    Returns:
        The parsed settings (``{"model": ..., "keys": {...}}``), or ``{}`` when the file is missing.

    Raises:
        ValueError: The file is not valid JSON, not a JSON object, or ``keys`` is malformed.
    """
    path = Path(path) if path is not None else AUTH_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        warnings.warn(
            f"{path} is readable by group/others ({stat.filemode(mode)}); expected 0600. "
            "Saving settings rewrites the file with safe permissions.",
            stacklevel=2,
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(data).__name__}")
    if "model" in data and not isinstance(data["model"], str):
        raise ValueError(f'{path}: "model" must be a string')
    keys = data.get("keys", {})
    if not isinstance(keys, dict):
        raise ValueError(f'{path}: "keys" must be a JSON object')
    data["keys"] = {str(name): str(value) for name, value in keys.items()}
    return data


def apply_to_environ(settings: dict[str, Any]) -> None:
    """Merge the stored keys into ``os.environ``; the real environment always wins."""
    for name, value in settings.get("keys", {}).items():
        os.environ.setdefault(name, value)


def save(
    model: str | None = None, keys: dict[str, str] | None = None, path: Path | None = None
) -> dict[str, Any]:
    """Update the settings file, changing only what was passed.

    Args:
        model: New model id, or ``None`` to keep the stored one.
        keys: Env var values that changed. An empty value removes the stored key. Never pass a
            masked or placeholder value back: only real values the user entered.
        path: File to write; defaults to :data:`AUTH_FILE`.

    Returns:
        The full settings dict now on disk.
    """
    path = Path(path) if path is not None else AUTH_FILE
    data: dict[str, Any]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a loose mode was already reported by load()
        try:
            data = load(path)
        except (OSError, ValueError):
            data = {}  # unreadable or corrupt: start over, the write below fixes it
    if model:
        data["model"] = model
    stored = dict(data.get("keys", {}))
    for name, value in (keys or {}).items():
        if value:
            stored[name] = value
        else:
            stored.pop(name, None)
    data["keys"] = stored
    _write(path, data)
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    """Write *data* atomically: temp file in the same dir, then ``os.replace``, mode ``0600``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
