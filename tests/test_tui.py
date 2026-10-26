import asyncio
import json

import lmdk.completion
from lmdk.datatypes import RawResponse
from lmdk.provider import Provider
from pydantic import BaseModel
from rich.text import Text
from textual.widgets import Collapsible, Static, TextArea

from markov_agent import tui
from markov_agent.skill_state import StepEvent


class State(BaseModel):
    plan: str
    notes: str = ""


def fake_run(state, observation, **_):
    s1 = {**state, "plan": "- [ ] look"}
    yield StepEvent(t=0, state=s1, patch={}, action="ls", observation=observation)
    output = "(exit 0)\nfile"
    yield StepEvent(t=1, state=s1, patch={}, action=None, reply="done", observation=output)


def broken_run(**_):
    raise RuntimeError("provider down")
    yield


def test_field_diff_marks_removed_and_added_lines():
    assert tui.field_diff("a\nb", "a\nc").plain == "  a\n- b\n+ c"
    assert tui.field_diff("unset", "x").plain == "+ x"


def _play(monkeypatch, fake, check):
    monkeypatch.setattr(tui, "run", fake)

    async def scenario():
        app = tui.MarkovApp("fake:model", "be good", State)
        async with app.run_test() as pilot:
            app.query_one(TextArea).text = "do it"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await check(app, pilot)

    asyncio.run(scenario())


def test_turn_ends_on_reply_and_scrubbing_shows_the_patch(monkeypatch):
    async def check(app, pilot):
        assert app.request == "do it"
        assert app.query_one("#output").border_title == "REPLY"
        assert app.status == "your turn"
        assert app.query_one("#observation").border_title == "OBSERVATION O₁ ← a₀"
        assert app.query_one("#cause").display
        assert not app.query_one(TextArea).disabled
        await pilot.press("pageup")
        assert app.query_one("#output").border_title == "ACTION a₀"
        assert app.query_one("#observation").border_title == "OBSERVATION O₀ ← user"
        assert not app.query_one("#cause").display
        plan = app.query_one("#field-plan", Collapsible)
        assert not plan.collapsed
        assert plan.title == "plan · patched"
        assert app.query_one("#field-notes", Collapsible).title == "notes · unchanged"

    _play(monkeypatch, fake_run, check)


def test_errors_keep_the_session_alive(monkeypatch):
    async def check(app, _):
        assert app.status == "error: provider down"
        assert not app.query_one(TextArea).disabled

    _play(monkeypatch, broken_run, check)


def test_prompt_wraps_grows_and_keeps_newlines(monkeypatch):
    monkeypatch.setattr(tui, "run", fake_run)

    async def scenario():
        app = tui.MarkovApp("fake:model", "be good", State)
        async with app.run_test() as pilot:
            box = app.query_one(TextArea)
            await pilot.press("a")
            await pilot.press("shift+enter")
            await pilot.press("b")
            assert box.text == "a\nb"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.request == "a\nb"
            assert box.text == ""
            box.text = "word " * 200
            await pilot.pause()
            assert box.size.height == 5  # grows up to 5 text lines, then scrolls

    asyncio.run(scenario())


def _timeline(app) -> str:
    content = app.query_one("#timeline", Static).content
    assert isinstance(content, Text)
    return content.plain


class FakeProvider(Provider):
    """One action then one reply, reporting fixed token usage on every call."""

    calls = 0

    @classmethod
    def _build_auth_headers(cls, credentials):
        return {}

    @classmethod
    def _send_completion_request(cls, request, credentials):
        cls.calls += 1
        reply = cls.calls % 2 == 0
        step = {"patch": {}, "kind": "reply" if reply else "action", "content": "done"}
        return RawResponse(
            content=json.dumps(step), input_tokens=1000 + cls.calls, output_tokens=10 * cls.calls
        )


def test_prompt_tokens_per_step_reach_the_ui(monkeypatch):
    FakeProvider.calls = 0
    monkeypatch.setattr(lmdk.completion, "resolve_model", lambda model: ("fake", "m", FakeProvider))

    async def scenario():
        app = tui.MarkovApp("fake:model", "be good", State)
        async with app.run_test() as pilot:
            app.query_one(TextArea).text = "do it"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert [(s[3], s[4]) for s in app.steps] == [(1001, 10), (1002, 20)]
            assert "1.0k in / 20 out" in _timeline(app)
            await pilot.press("pageup")
            timeline = _timeline(app)
            assert "1.0k in / 10 out" in timeline  # that step's own counts
            assert "2.0k in / 30 out" in timeline  # cumulative turn total

    asyncio.run(scenario())
