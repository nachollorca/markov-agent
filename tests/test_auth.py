import json
import os
import stat
import warnings

import pytest

from markov_agent import auth


def test_load_missing_file_returns_empty(tmp_path):
    assert auth.load(tmp_path / "auth.json") == {}


def test_load_corrupt_json_raises_clear_error(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{not json")
    os.chmod(path, 0o600)
    with pytest.raises(ValueError, match="not valid JSON"):
        auth.load(path)


def test_load_non_object_json_raises_clear_error(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("[1, 2]")
    os.chmod(path, 0o600)
    with pytest.raises(ValueError, match="must contain a JSON object"):
        auth.load(path)


def test_load_bad_keys_entry_raises_clear_error(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text('{"keys": [1]}')
    os.chmod(path, 0o600)
    with pytest.raises(ValueError, match='"keys" must be a JSON object'):
        auth.load(path)


def test_load_loose_permissions_warns_but_still_loads(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"keys": {"MISTRAL_API_KEY": "k"}}))
    os.chmod(path, 0o644)
    with pytest.warns(UserWarning, match="readable by group/others"):
        data = auth.load(path)
    assert data["keys"] == {"MISTRAL_API_KEY": "k"}


def test_load_returns_model_and_keys(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"model": "mistral:x", "keys": {"MISTRAL_API_KEY": "k"}}))
    os.chmod(path, 0o600)
    assert auth.load(path) == {"model": "mistral:x", "keys": {"MISTRAL_API_KEY": "k"}}


def test_load_defaults_to_auth_file(monkeypatch, tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"model": "m"}))
    os.chmod(path, 0o600)
    monkeypatch.setattr(auth, "AUTH_FILE", path)
    assert auth.load() == {"model": "m", "keys": {}}
    auth.save(model="m2")
    assert json.loads(path.read_text())["model"] == "m2"


def test_save_writes_0600_file_in_0700_dir_and_round_trips(tmp_path):
    path = tmp_path / "auth.json"
    auth.save(model="mistral:x", keys={"MISTRAL_API_KEY": "secret"}, path=path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert auth.load(path) == {"model": "mistral:x", "keys": {"MISTRAL_API_KEY": "secret"}}


def test_save_updates_only_what_changed_and_preserves_unrelated_keys(tmp_path):
    path = tmp_path / "auth.json"
    auth.save(
        model="mistral:x", keys={"MISTRAL_API_KEY": "one", "OPENAI_API_KEY": "keep"}, path=path
    )
    auth.save(keys={"MISTRAL_API_KEY": "two"}, path=path)
    data = auth.load(path)
    assert data["model"] == "mistral:x"  # untouched
    assert data["keys"] == {"MISTRAL_API_KEY": "two", "OPENAI_API_KEY": "keep"}


def test_save_empty_value_removes_the_stored_key(tmp_path):
    path = tmp_path / "auth.json"
    auth.save(keys={"MISTRAL_API_KEY": "one"}, path=path)
    auth.save(keys={"MISTRAL_API_KEY": ""}, path=path)
    assert auth.load(path)["keys"] == {}


def test_save_over_corrupt_file_starts_fresh(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("nope")
    auth.save(model="mistral:x", path=path)
    assert auth.load(path) == {"model": "mistral:x", "keys": {}}


def test_save_does_not_warn_about_loose_permissions(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"keys": {"A": "1"}}))
    os.chmod(path, 0o644)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        auth.save(model="m", path=path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600  # overwriting fixed it


def test_save_failure_cleans_up_the_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        auth.save(model="m", path=path)
    assert list(tmp_path.iterdir()) == []  # no temp file, no auth.json left behind


def test_apply_to_environ_never_overwrites_the_real_environment(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth.apply_to_environ({"keys": {"MISTRAL_API_KEY": "from-file", "OPENAI_API_KEY": "file-only"}})
    assert os.environ["MISTRAL_API_KEY"] == "from-env"
    assert os.environ["OPENAI_API_KEY"] == "file-only"
