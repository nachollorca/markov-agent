"""Terminal UI for SKILL.state agents: watch the state get patched step by step.

The screen mirrors what the model sees at step t: the state Σₜ on the left, where each field
shows the diff the patch makes to produce Σₜ₊₁, and on the right the observation Oₜ plus the
action or reply the model answered with. A timeline scrubs through past steps and the input
opens the session (the request) and continues it (the reply to each of the agent's replies).
"""

import argparse
import difflib
import json
import os
from typing import Any

from pydantic import BaseModel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Collapsible, Footer, Header, Input, Static
from textual.worker import get_current_worker

from .agent import DEFAULT_MODEL, INSTRUCTIONS, CodingAgentState
from .skill_state import StepEvent, initial_state, run

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_DIFF_STYLES = {"+": "green", "-": "red", " ": ""}


def _lines(value: Any) -> list[str]:
    if value == "unset":
        return []
    return (value if isinstance(value, str) else json.dumps(value, indent=2)).splitlines()


def field_diff(old: Any, new: Any) -> Text:
    """Line diff of one state field: removed lines red, added lines green."""
    lines = difflib.ndiff(_lines(old), _lines(new))
    # ndiff also emits "?" hint lines, which only make sense in monospace side by side
    return Text("\n").join(Text(line, _DIFF_STYLES[line[0]]) for line in lines if line[0] != "?")


class MarkovApp(App):
    """Run a SKILL.state agent one turn at a time, showing each step as state + patch."""

    CSS = """
    #main { height: 1fr; margin-top: 1; }
    #state, #step { width: 1fr; }
    #state { border: round $primary; }
    #observation { height: 2fr; border: round $secondary; }
    #cause { color: $accent; text-style: bold; border-bottom: solid $secondary; display: none; }
    #output { height: 1fr; border: round $accent; }
    #output.reply { border: heavy $success; }
    Collapsible { padding-bottom: 0; }
    Collapsible Contents { padding-top: 0; }
    Collapsible.patched { border-left: thick $warning; }
    #timeline { height: auto; max-height: 3; padding: 0 1; }
    Input { margin-bottom: 1; }
    """
    BINDINGS = [  # noqa: RUF012 (textual's own convention)
        # priority: pageup/pagedown must scrub even while typing in the input
        Binding("pageup", "scrub(-1)", "Previous step", priority=True),
        Binding("pagedown", "scrub(1)", "Next step", priority=True),
    ]

    def __init__(
        self,
        model: str,
        instructions: str,
        state_schema: type[BaseModel],
        state: dict | None = None,
    ):
        super().__init__()
        self.model = model
        self.instructions = instructions
        self.state_schema = state_schema
        self.state = initial_state(state_schema, state)
        self.request = ""
        self.turn = 0
        self.steps: list[tuple[int, dict, StepEvent]] = []  # (turn, state before the step, step)
        self.selected = -1  # selected step; following live while it is the last one
        self.status = "type your request below"

    def compose(self) -> ComposeResult:
        """Lay out a fixed panel per state field, so nothing moves between steps."""
        yield Header()
        with Horizontal(id="main"):
            with VerticalScroll(id="state"):
                with Collapsible(title="instructions (P)"):
                    yield Static(self.instructions, markup=False)
                for name in self.state_schema.model_fields:
                    with Collapsible(title=name, id=f"field-{name}"):
                        yield Static(id=f"value-{name}", markup=False)
            with Vertical(id="step"):
                with Vertical(id="observation"):
                    yield Static(id="cause", markup=False)  # the action whose output this is
                    with VerticalScroll():
                        yield Static(id="obs", markup=False)
                with VerticalScroll(id="output"):
                    yield Static(id="out", markup=False)
        yield Static(id="timeline")
        yield Input(placeholder="Your request…")
        yield Footer()

    def on_mount(self) -> None:
        """Show the initial state Σ₀ before anything ran."""
        self.title = f"markov · {self.model}"
        self._show_state(self.state, self.state, "STATE Σ₀")
        self._show_timeline()
        self.query_one(Input).focus()

    def on_input_submitted(self, message: Input.Submitted) -> None:
        """The first message is the request, every later one is the next turn's observation."""
        text = message.value.strip()
        if not text:
            return
        message.input.clear()
        observation = text
        if not self.request:
            self.request, observation = text, "none"
            self.sub_title = text
            message.input.placeholder = "Your reply…"
        self.turn += 1
        self._set_busy("working…", busy=True)
        self._run_turn(observation)

    @work(thread=True, exclusive=True)
    def _run_turn(self, observation: str) -> None:
        """Consume `run` off the UI thread and hand every step to the UI as it arrives."""
        worker = get_current_worker()
        status = "step limit reached, reply to continue"
        try:
            for event in run(
                model=self.model,
                instructions=self.instructions,
                request=self.request,
                state_schema=self.state_schema,
                state=self.state,
                observation=observation,
            ):
                if worker.is_cancelled:  # app is quitting
                    return
                self.call_from_thread(self._add_step, event)
                if event.reply:
                    status = "your turn"
        except Exception as error:  # keep the session: the state survives, the user can retry
            status = f"error: {error}"
        if not worker.is_cancelled:
            self.call_from_thread(self._set_busy, status, busy=False)

    def _add_step(self, event: StepEvent) -> None:
        following = self.selected == len(self.steps) - 1
        self.steps.append((self.turn, self.state, event))
        self.state = event.state
        if following:
            self.selected = len(self.steps) - 1
        self._show_step()

    def _set_busy(self, status: str, busy: bool) -> None:
        self.status = status
        box = self.query_one(Input)
        box.disabled = busy
        if not busy:
            box.focus()
        self._show_timeline()

    def action_scrub(self, delta: int) -> None:
        """Move the selection through past steps; reaching the last one follows live again."""
        if self.steps:
            self.selected = max(0, min(len(self.steps) - 1, self.selected + delta))
            self._show_step()

    def _show_step(self) -> None:
        _, before, event = self.steps[self.selected]
        t, t1 = str(event.t).translate(_SUB), str(event.t + 1).translate(_SUB)
        self._show_state(before, event.state, f"STATE Σ{t} → Σ{t1}")
        # Oₜ is the output of the previous step's action, or the user's message at a turn start
        cause = self.steps[self.selected - 1][2].action if event.t else None
        source = f"a{str(event.t - 1).translate(_SUB)}" if cause else "user"
        self.query_one("#observation").border_title = f"OBSERVATION O{t} ← {source}"
        cause_box = self.query_one("#cause", Static)
        cause_box.display = bool(cause)
        cause_box.update(f"$ {cause}")
        self.query_one("#obs", Static).update(event.observation or "none")
        output = self.query_one("#output")
        output.set_class(bool(event.reply), "reply")
        output.border_title = "REPLY" if event.reply else f"ACTION a{t}"
        self.query_one("#out", Static).update(event.reply or event.action or "")
        self._show_timeline()

    def _show_state(self, before: dict, after: dict, title: str) -> None:
        self.query_one("#state").border_title = title
        for name in self.state_schema.model_fields:
            old, new = before.get(name), after.get(name)
            changed = old != new
            label = "patched" if changed else "unset" if new == "unset" else "unchanged"
            box = self.query_one(f"#field-{name}", Collapsible)
            box.title = f"{name} · {label}"
            box.collapsed = not changed
            box.set_class(changed, "patched")
            value = field_diff(old, new) if changed else Text("\n".join(_lines(new)), "dim")
            self.query_one(f"#value-{name}", Static).update(value)

    def _show_timeline(self) -> None:
        # shortcut: one glyph per step wraps after ~3 lines of steps, upgrade to a scrolling strip
        line = Text()
        for i, (_, _, event) in enumerate(self.steps):
            if event.t == 0 and i:
                line.append("│", "dim")
            line.append("◉" if i == self.selected else "○", "bold" if i == self.selected else "dim")
        if self.steps:
            turn, _, event = self.steps[self.selected]
            line.append(f"  turn {turn} · step {event.t}")
        line.append(f"   {self.status}", "italic")
        self.query_one("#timeline", Static).update(line)


def main() -> None:
    """Entry point of the ``markov`` command."""
    parser = argparse.ArgumentParser(description="Run the coding agent in a terminal UI.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Language model to use.")
    parser.add_argument("--working-dir", default=os.getcwd(), help="Project directory.")
    args = parser.parse_args()

    if os.getenv("LMDK_TELEMETRY"):
        import logfire

        logfire.configure(
            token=os.environ["LOGFIRE_TOKEN"],
            service_name="skill-state-coding-agent",
            scrubbing=False,
            send_to_logfire=True,
            console=False,  # console output would draw over the TUI
        )

    state = {"working_dir": os.path.abspath(args.working_dir)}
    MarkovApp(args.model, INSTRUCTIONS, CodingAgentState, state).run()
