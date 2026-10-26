import asyncio

from pydantic import BaseModel
from textual.widgets import Collapsible, TextArea

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
