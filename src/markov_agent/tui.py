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
import pkgutil
import sys
import warnings
from typing import Any

import lmdk.providers
from lmdk import CompletionObserver, observe
from lmdk.provider import load_provider, resolve_model
from pydantic import BaseModel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import CommandPalette
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Footer, Header, Input, Label, Static, TextArea
from textual.worker import get_current_worker

from . import config
from .agent import DEFAULT_MODEL, INSTRUCTIONS, CodingAgentState
from .skill_state import StepEvent, initial_state, run

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_DIFF_STYLES = {"+": "green", "-": "red", " ": ""}


def _lines(value: Any) -> list[str]:
    if value == "unset":
        return []
    return (value if isinstance(value, str) else json.dumps(value, indent=2)).splitlines()


def _count(tokens: int | None) -> str:
    return "?" if tokens is None else f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def step_tokens(obs: CompletionObserver) -> tuple[int | None, int | None]:
    """Prompt and output tokens of the last completion: the step `run` just yielded."""
    if not obs.records:
        return None, None
    response = obs.records[-1].response
    return response.input_tokens, response.output_tokens


def field_diff(old: Any, new: Any) -> Text:
    """Line diff of one state field: removed lines red, added lines green."""
    lines = difflib.ndiff(_lines(old), _lines(new))
    # ndiff also emits "?" hint lines, which only make sense in monospace side by side
    return Text("\n").join(Text(line, _DIFF_STYLES[line[0]]) for line in lines if line[0] != "?")


def mask(secret: str) -> str:
    """Mask a secret for display, keeping only the first 3 and last 4 characters."""
    return f"{secret[:3]}…{secret[-4:]}" if len(secret) > 7 else "…"


def key_status(name: str, stored: dict) -> tuple[str | None, str | None]:
    """Where the current value of env var *name* comes from, and its mask (``None`` when unset)."""
    value = os.getenv(name)
    if value:
        return "env", mask(value)
    value = stored.get("keys", {}).get(name)
    if value:
        return "config.json", mask(value)
    return None, None


def provider_key_envs() -> list[tuple[str, tuple[str, ...]]]:
    """Every ``(provider, required env vars)`` pair lmdk knows, alphabetical; keyless skipped."""
    pairs = []
    for module in pkgutil.iter_modules(lmdk.providers.__path__):
        if module.name.startswith("_"):
            continue
        try:
            provider = load_provider(module.name)
        except (ImportError, AttributeError):
            continue
        required = provider.required_env
        names = (required,) if isinstance(required, str) else tuple(required)
        if names:
            pairs.append((module.name, names))
    return sorted(pairs)


class SettingsScreen(ModalScreen):
    """Pick the model and enter provider API keys; saved to ``~/.markov/config.json`` on Save."""

    BINDINGS = [Binding("escape", "dismiss", "Close")]  # noqa: RUF012 (textual's own convention)

    def __init__(self, model: str, stored: dict):
        super().__init__()
        self._model = model
        self._stored = stored

    def compose(self) -> ComposeResult:
        """One model input plus one masked password input per known provider env var."""
        with VerticalScroll(id="settings-dialog"):
            yield Static("Settings", id="settings-title")
            yield Label("Model (provider:model-id)")
            yield Input(self._model, id="model")
            yield Static(id="settings-error")
            for provider, names in provider_key_envs():
                yield Label(provider, classes="provider")
                for name in names:
                    source, masked = key_status(name, self._stored)
                    hint = f"  {masked} ({source})" if masked else "  (not set)"
                    yield Label(f"{name}{hint}", id=f"keylabel-{name}")
                    # never pre-filled: empty means "unchanged", so no mask is saved back
                    placeholder = "leave empty to keep current" if masked else ""
                    yield Input(password=True, id=f"key-{name}", placeholder=placeholder)
            with Horizontal(id="settings-buttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the model input so the dialog is usable from the keyboard."""
        self.query_one("#model", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Cancel dismisses without writing; Save validates and dismisses with the new settings."""
        if event.button.id == "cancel":
            self.dismiss(None)
        else:
            self.request_save()

    def request_save(self) -> None:
        """Validate the model through lmdk, collect the entered keys, dismiss with them.

        An invalid model shows the error inline and keeps the dialog open instead of crashing.
        """
        model = self.query_one("#model", Input).value.strip()
        error_box = self.query_one("#settings-error", Static)
        try:
            if not model:
                raise ValueError("model is empty")
            resolve_model(model)
        except Exception as error:  # model ids only, never key values
            error_box.update(f"cannot use model: {error}")
            error_box.display = True
            return
        error_box.display = False
        keys = {}
        for _, names in provider_key_envs():
            for name in names:
                value = self.query_one(f"#key-{name}", Input).value.strip()
                if value:
                    keys[name] = value
        self.dismiss({"model": model, "keys": keys})


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
    TextArea { height: auto; max-height: 7; margin-bottom: 1; }  /* 5 text lines + 2 border rows */
    SettingsScreen { align: center middle; }
    #settings-dialog {
        width: 90%; max-width: 110; height: auto; max-height: 95%;
        border: round $primary; background: $surface; padding: 1 2;
    }
    #settings-title { text-style: bold; }
    #settings-dialog Label { padding-top: 1; color: $text-muted; }
    #settings-dialog Label.provider { color: $accent; text-style: bold; }
    #settings-error { color: $error; display: none; }
    #settings-buttons { height: auto; padding-top: 1; align-horizontal: right; }
    """
    BINDINGS = [  # noqa: RUF012 (textual's own convention)
        # priority: pageup/pagedown must scrub even while typing in the prompt
        Binding("pageup", "scrub(-1)", "Previous step", priority=True),
        Binding("pagedown", "scrub(1)", "Next step", priority=True),
        # priority: TextArea would otherwise swallow enter and insert a newline
        Binding("enter", "submit", "Send", priority=True),
        # shift+enter, or ctrl+j where the terminal cannot tell the two apart
        Binding("shift+enter,ctrl+j", "newline", "New line", show=False),
        # ctrl+s is free: neither Input nor TextArea binds it (only their ctrl+shift+ combos)
        Binding("ctrl+s", "settings", "Settings"),
    ]

    def __init__(
        self,
        model: str,
        instructions: str,
        state_schema: type[BaseModel],
        state: dict | None = None,
        theme: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.stored_theme = theme  # from config.json; saved on every change after startup
        self.instructions = instructions
        self.state_schema = state_schema
        self.state = initial_state(state_schema, state)
        self.request = ""
        self.turn = 0
        self.busy = False  # a turn is running: the input is disabled and settings stay closed
        # (turn, state before the step, step, prompt tokens, output tokens)
        self.steps: list[tuple[int, dict, StepEvent, int | None, int | None]] = []
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
        yield TextArea(placeholder="Your request…")
        yield Footer()

    def on_mount(self) -> None:
        """Show the initial state Σ₀ before anything ran."""
        if self.stored_theme in self.available_themes:
            self.theme = self.stored_theme
        self.title = f"markov · {self.model}"
        self._show_state(self.state, self.state, "STATE Σ₀")
        self._show_timeline()
        self.query_one(TextArea).focus()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Keep the app's priority bindings from stealing keys from the command palette."""
        # enter/pageup/pagedown belong to the palette's own input and list
        palette = isinstance(self.screen, CommandPalette)
        return not (palette and action in ("submit", "newline", "scrub"))

    def watch_theme(self, theme: str) -> None:
        """Persist a theme picked from the command palette, so the next run starts with it."""
        if theme == self.stored_theme:
            return  # startup restore, not a choice by the user
        try:
            config.save(theme=theme)
        except OSError as error:
            self.status = f"error saving theme: {error}"
            return
        self.stored_theme = theme

    def action_submit(self) -> None:
        """The first message is the request, every later one is the next turn's observation."""
        if isinstance(self.screen, SettingsScreen):  # enter confirms the dialog instead
            self.screen.request_save()
            return
        box = self.query_one(TextArea)
        text = box.text.strip()
        if not text:
            return
        box.clear()
        observation = text
        if not self.request:
            self.request, observation = text, "none"
            self.sub_title = text
            box.placeholder = "Your reply…"
        self.turn += 1
        self._set_busy("working…", busy=True)
        self._run_turn(observation)

    def action_newline(self) -> None:
        """Insert a line break instead of sending."""
        if isinstance(self.screen, SettingsScreen):
            return
        self.query_one(TextArea).insert("\n")

    def action_settings(self) -> None:
        """Open the settings screen; closed while a turn runs or the screen is already open."""
        if self.busy or isinstance(self.screen, SettingsScreen):
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # main() already reported config.json problems
            try:
                stored = config.load()
            except (OSError, ValueError):
                stored = {}
        self.push_screen(SettingsScreen(self.model, stored), self._apply_settings)

    def _apply_settings(self, result: dict | None) -> None:
        """Persist and apply saved settings: model on the app, keys in the env and on disk."""
        if not result:
            return  # cancelled
        try:
            config.save(model=result["model"], keys=result["keys"])
        except OSError as error:  # paths only, never key values
            self.status = f"error saving settings: {error}"
        else:
            self.model = result["model"]
            self.title = f"markov · {self.model}"
            os.environ.update(result["keys"])  # the next turn uses them without a restart
            self.status = "settings saved"
        self._show_timeline()

    @work(thread=True, exclusive=True)
    def _run_turn(self, observation: str) -> None:
        """Consume `run` off the UI thread and hand every step to the UI as it arrives."""
        worker = get_current_worker()
        status = "step limit reached, reply to continue"
        try:
            # contextvars-scoped, so the block must wrap the loop in this very thread
            with observe() as obs:
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
                    self.call_from_thread(self._add_step, event, *step_tokens(obs))
                    if event.reply:
                        status = "your turn"
        except Exception as error:  # keep the session: the state survives, the user can retry
            status = f"error: {error}"
        if not worker.is_cancelled:
            self.call_from_thread(self._set_busy, status, busy=False)

    def _add_step(
        self, event: StepEvent, prompt_tokens: int | None = None, output_tokens: int | None = None
    ) -> None:
        following = self.selected == len(self.steps) - 1
        self.steps.append((self.turn, self.state, event, prompt_tokens, output_tokens))
        self.state = event.state
        if following:
            self.selected = len(self.steps) - 1
        self._show_step()

    def _set_busy(self, status: str, busy: bool) -> None:
        self.status = status
        self.busy = busy
        box = self.query_one(TextArea)
        box.disabled = busy
        if not busy:
            box.focus()
        self._show_timeline()

    def action_scrub(self, delta: int) -> None:
        """Move the selection through past steps; reaching the last one follows live again."""
        if isinstance(self.screen, SettingsScreen):
            return
        if self.steps:
            self.selected = max(0, min(len(self.steps) - 1, self.selected + delta))
            self._show_step()

    def _show_step(self) -> None:
        _, before, event, _, _ = self.steps[self.selected]
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

    def _step_glyphs(self) -> Text:
        """One glyph per step, with a bar where a new turn starts."""
        # shortcut: one glyph per step wraps after ~3 lines of steps, upgrade to a scrolling strip
        line = Text()
        for i, (_, _, event, _, _) in enumerate(self.steps):
            if event.t == 0 and i:
                line.append("│", "dim")
            line.append("◉" if i == self.selected else "○", "bold" if i == self.selected else "dim")
        return line

    def _turn_total(self, turn: int) -> str:
        """Cumulative tokens of *turn*: flat step counts mean the prompt is not growing."""
        steps = [s for s in self.steps if s[0] == turn]
        return (
            f" · turn {_count(sum(s[3] or 0 for s in steps))} in / "
            f"{_count(sum(s[4] or 0 for s in steps))} out"
        )

    def _show_timeline(self) -> None:
        line = self._step_glyphs()
        if self.steps:
            turn, _, event, prompt_tokens, output_tokens = self.steps[self.selected]
            line.append(f"  turn {turn} · step {event.t}")
            if prompt_tokens is not None:
                line.append(f" · {_count(prompt_tokens)} in / {_count(output_tokens)} out")
                line.append(self._turn_total(turn), "dim")
        line.append(f"   {self.status}", "italic")
        self.query_one("#timeline", Static).update(line)


def resolve_startup_model(flag: str | None, settings: dict) -> str:
    """Model for this run: an explicit ``--model`` flag wins, then config.json, then the default."""
    return flag or settings.get("model") or DEFAULT_MODEL


def main() -> None:
    """Entry point of the ``markov`` command."""
    parser = argparse.ArgumentParser(description="Run the coding agent in a terminal UI.")
    parser.add_argument("--model", default=None, help="Model to use; overrides config.json.")
    parser.add_argument("--working-dir", default=os.getcwd(), help="Project directory.")
    args = parser.parse_args()

    try:
        settings = config.load()
    except (OSError, ValueError) as error:  # corrupt config.json: say it plainly, keep going
        print(f"markov: {error}", file=sys.stderr)
        settings = {}
    config.apply_to_environ(settings)  # before the first completion; the real env wins
    model = resolve_startup_model(args.model, settings)

    if os.getenv("LMDK_TELEMETRY"):
        import logfire

        logfire.configure(
            token=os.environ["LOGFIRE_TOKEN"],
            service_name="skill-state-coding-agent",
            scrubbing=False,
            send_to_logfire=True,
            console=False,  # console output would draw over the TUI
        )

    os.chdir(args.working_dir)  # commands run in a shell that inherits this cwd
    MarkovApp(model, INSTRUCTIONS, CodingAgentState, theme=settings.get("theme")).run()
