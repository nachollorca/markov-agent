import asyncio
import json
import os
import stat
import threading

import pytest
from pydantic import BaseModel
from textual.widgets import Button, Input, Label, Static, TextArea

from markov_agent import config, tui
from markov_agent.skill_state import StepEvent


class State(BaseModel):
    plan: str
    notes: str = ""


@pytest.fixture
def config_file(monkeypatch, tmp_path):
    """Isolate ~/.markov/config.json and clear every provider key from the real environment."""
    for _, names in tui.provider_key_envs():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", path)
    return path


def test_mask_shows_only_the_edges():
    assert tui.mask("sk-ant-abcdef1234") == "sk-…1234"
    assert tui.mask("short") == "…"


def test_provider_key_envs_lists_every_keyed_provider():
    pairs = dict(tui.provider_key_envs())
    assert pairs["mistral"] == ("MISTRAL_API_KEY",)
    assert pairs["vertex"] == ("VERTEX_API_KEY", "GCP_PROJECT_ID")
    assert "local" not in pairs and "laya" not in pairs  # no credentials required
    assert tui.provider_key_envs() == sorted(tui.provider_key_envs())


def test_resolve_startup_model_precedence():
    assert tui.resolve_startup_model("flag:model", {"model": "file:model"}) == "flag:model"
    assert tui.resolve_startup_model(None, {"model": "file:model"}) == "file:model"
    assert tui.resolve_startup_model(None, {}) == tui.DEFAULT_MODEL


def test_ctrl_s_opens_settings_and_escape_cancels_without_writing(config_file):
    async def scenario():
        app = tui.MarkovApp("mistral:mistral-large-4", "be good", State)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, tui.SettingsScreen)
            screen = app.screen
            assert screen.query_one("#model", Input).value == "mistral:mistral-large-4"
            # never pre-filled with the real or a masked value
            assert screen.query_one("#key-MISTRAL_API_KEY", Input).value == ""
            label = str(screen.query_one("#keylabel-MISTRAL_API_KEY", Label).render())
            assert "MISTRAL_API_KEY" in label and "(not set)" in label
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, tui.SettingsScreen)
            assert app.model == "mistral:mistral-large-4"
            assert not config_file.exists()  # cancel wrote nothing

    asyncio.run(scenario())


def test_save_persists_and_applies_model_and_keys(config_file):
    async def scenario():
        app = tui.MarkovApp("mistral:old", "be good", State)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+s")
            await pilot.pause()
            screen = app.screen
            screen.query_one("#model", Input).value = "anthropic:claude-sonnet-4-5"
            screen.query_one("#key-ANTHROPIC_API_KEY", Input).value = "sk-ant-test-1234"
            screen.query_one("#save", Button).press()
            await pilot.pause()
            assert not isinstance(app.screen, tui.SettingsScreen)
            assert app.model == "anthropic:claude-sonnet-4-5"
            assert app.title == "markov · anthropic:claude-sonnet-4-5"
            assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-1234"
            assert app.status == "settings saved"
            assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
            data = json.loads(config_file.read_text())
            assert data == {
                "model": "anthropic:claude-sonnet-4-5",
                "keys": {"ANTHROPIC_API_KEY": "sk-ant-test-1234"},
            }

    asyncio.run(scenario())


def test_enter_in_the_dialog_saves(config_file):
    async def scenario():
        app = tui.MarkovApp("mistral:old", "be good", State)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+s")
            await pilot.pause()
            app.screen.query_one("#model", Input).value = "openai:gpt-5"
            await pilot.press("enter")  # the app's priority enter confirms the dialog
            await pilot.pause()
            assert not isinstance(app.screen, tui.SettingsScreen)
            assert app.model == "openai:gpt-5"
            assert json.loads(config_file.read_text())["model"] == "openai:gpt-5"

    asyncio.run(scenario())


def test_enter_in_the_command_palette_is_not_stolen_by_submit(config_file):
    async def scenario():
        app = tui.MarkovApp("mistral:mistral-large-4", "be good", State)
        async with app.run_test() as pilot:
            app.query_one(TextArea).text = "do it"
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press(*"theme")  # goes to the palette's input, not the prompt
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            # the palette handled enter; the app's priority submit did not fire
            assert not app.busy and app.turn == 0
            assert app.query_one(TextArea).text == "do it"

    asyncio.run(scenario())


def test_invalid_model_shows_the_error_inline_and_stays_open(config_file):
    async def scenario():
        app = tui.MarkovApp("mistral:old", "be good", State)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+s")
            await pilot.pause()
            screen = app.screen
            screen.query_one("#model", Input).value = "not-a-model"
            screen.query_one("#save", Button).press()
            await pilot.pause()
            assert isinstance(app.screen, tui.SettingsScreen)  # not dismissed, session alive
            error = screen.query_one("#settings-error", Static)
            assert error.display
            assert "cannot use model" in str(error.render())
            assert not config_file.exists()
            screen.query_one("#model", Input).value = "mistral:mistral-large-4"
            screen.query_one("#save", Button).press()
            await pilot.pause()
            assert not isinstance(app.screen, tui.SettingsScreen)
            assert app.model == "mistral:mistral-large-4"

    asyncio.run(scenario())


def test_existing_keys_shown_masked_with_their_source(monkeypatch, config_file):
    monkeypatch.setenv("MISTRAL_API_KEY", "REDACTED")
    config_file.write_text(json.dumps({"keys": {"OPENAI_API_KEY": "sk-openai-xyz99999"}}))
    os.chmod(config_file, 0o600)

    async def scenario():
        app = tui.MarkovApp("mistral:mistral-large-4", "be good", State)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+s")
            await pilot.pause()
            screen = app.screen
            env_label = str(screen.query_one("#keylabel-MISTRAL_API_KEY", Label).render())
            file_label = str(screen.query_one("#keylabel-OPENAI_API_KEY", Label).render())
            assert "RED…CTED (env)" in env_label
            assert "sk-…9999 (config.json)" in file_label
            # the password inputs stay empty, so no mask is ever stored back on save
            assert screen.query_one("#key-MISTRAL_API_KEY", Input).value == ""
            assert screen.query_one("#key-OPENAI_API_KEY", Input).value == ""

    asyncio.run(scenario())


def test_ctrl_s_stays_closed_while_a_turn_runs(monkeypatch, config_file):
    release = threading.Event()

    def blocking_run(state, observation, **_):
        release.wait(10)
        yield StepEvent(
            t=0,
            state={**state, "plan": "p"},
            patch={},
            action=None,
            reply="done",
            observation=observation,
        )

    monkeypatch.setattr(tui, "run", blocking_run)

    async def scenario():
        app = tui.MarkovApp("mistral:mistral-large-4", "be good", State)
        async with app.run_test() as pilot:
            app.query_one(TextArea).text = "do it"
            await pilot.press("enter")
            await pilot.pause()
            assert app.busy
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert not isinstance(app.screen, tui.SettingsScreen)
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.busy
            await pilot.press("ctrl+s")  # idle again: it opens
            await pilot.pause()
            assert isinstance(app.screen, tui.SettingsScreen)

    asyncio.run(scenario())
