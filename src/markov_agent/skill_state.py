"""SKILL.state (arXiv:2608.26263) core implementation.

Prompt stays O(1) instead of growing with the conversation.
At each step t, the model sees only three things:
    1. P: The immutable skill spec
    2. Σₜ: The current state in the given per-domain structure
    3. Oₜ: The latest observation

It returns:
    - A state update (patch) that produces Σₜ₊₁
    - And either an action that will produce Oₜ₊₁, or a reply that ends the turn
      and hands control back to the user
"""

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from lmdk import complete, render_template
from pydantic import BaseModel, Field, create_model, model_validator

PROMPT_TEMPLATE = Path(__file__).parent / "prompt_template.jinja"  # ships with the package
_PATCH_KEYWORDS = ("unchanged", "unset")  # state-patch markers, never a real action or reply
BASH_TIMEOUT = 100  # seconds to wait for the bash
MAX_BASH_OUTPUT = 100_000  # maximum chars of observation shown to the model


def yolo_bash(command: str) -> str:
    """Run a bash command and return it alongside its exit code, stdout and stderr."""
    # ! Probably needs a white/black- list and / or a sandbox
    proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT)
    output = (proc.stdout + proc.stderr).strip() or "(no output)"
    # The model never sees its own previous action, so the observation carries it
    # Could also be `{{ ACTION }}` variable in the `prompt_template.jinja`
    truncated = ""
    if len(output) > MAX_BASH_OUTPUT:
        truncated = f"\n[The output has been truncated at the maximum {MAX_BASH_OUTPUT} characters]"
    return f"$ {command}\n(exit {proc.returncode})\n{output[:MAX_BASH_OUTPUT]}{truncated}"


def update_state(state: dict, patch: dict) -> dict:
    """Recursively merge *patch* (the state change made by the agent) into *state*."""
    merged = dict(state)
    for key, value in patch.items():
        if value == "unchanged":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = update_state(merged[key], value)
        else:
            merged[key] = value
    return merged


def derive_patch_schema(state_schema: type[BaseModel]) -> type[BaseModel]:
    """Derive from the domain-defined schema a patch schema the model can emit.

    Adds to each field the possibility to be "unchanged" (default) or "unset".
    """
    fields: dict[str, Any] = {}
    for name, field in state_schema.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            ann = derive_patch_schema(ann)
        patch_ann = ann | Literal["unchanged", "unset"]
        fields[name] = (patch_ann, Field(default="unchanged", description=field.description))
    return create_model(f"Patch_{state_schema.__name__}", **fields)


class _Step(BaseModel):
    """One step of the loop: mutate the state, then either act or hand the turn back."""

    action: str = Field(
        default="",
        description="A shell command to run now. Its output becomes the next observation. "
        "Fill it when the step is yours to execute, and leave `reply` empty.",
    )
    reply: str = Field(
        default="",
        description="Text for the user: the result, a question, or a blocker. Their answer "
        "becomes the next observation. Fill it when only they can decide or when the task is "
        "done, and leave `action` empty.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "_Step":
        # The patch keyword sometimes lands in the idle field instead of ""
        self.action = "" if self.action in _PATCH_KEYWORDS else self.action
        self.reply = "" if self.reply in _PATCH_KEYWORDS else self.reply
        if bool(self.action) == bool(self.reply):
            raise ValueError("fill exactly one of 'action' or 'reply', leave the other empty")
        return self


def derive_schema(state_schema: type[BaseModel]) -> type[BaseModel]:
    """Combine the patch schema with the action/reply fields the model answers with."""
    patch_schema = derive_patch_schema(state_schema)
    return create_model(
        f"Step{state_schema.__name__}",
        __base__=_Step,
        patch=(
            patch_schema,
            Field(..., description="State changes only, never `action` or `reply`."),
        ),
    )


def initial_state(state_schema: type[BaseModel], state: dict | None = None) -> dict:
    """Seed the state: required fields start as "unset", optional fields at their default."""
    defaults = {
        name: ("unset" if field.is_required() else field.get_default(call_default_factory=True))
        for name, field in state_schema.model_fields.items()
    }
    return {**defaults, **(state or {})}


class StepEvent(BaseModel):
    """Representation of model state, action and observation at time t."""

    t: int
    state: dict
    patch: dict
    action: str | None
    reply: str | None = None
    observation: str | None = None


def run(
    model: str,
    instructions: str,
    request: str,
    state_schema: type[BaseModel],
    state: dict | None = None,
    observation: str = "none",
    max_steps: int = 50,
) -> Iterator[StepEvent]:
    """Execute one turn, until the model replies to the user or *max_steps* is hit.

    Args:
        model: Provider-prefixed model identifier, as in ``lmdk.complete``.
        instructions: The immutable procedural specification (P in the paper).
        state_schema: Domain-defined structure of the state (Σ in the paper) for this task.
        request: The request that started the session. Stable across turns.
        state: Initial execution state (Σ₀). Empty by default, can be used to start from checkpoint.
        observation: The observation the turn starts from (O₀): the user's reply to the previous
            turn, or "none" on the first one.
        max_steps: Horizon cap, so a looping model cannot burn tokens forever.

    Yields:
        One ``StepEvent`` per step, carrying the post-patch state, the patch itself,
        the action proposed and the observation it was conditioned on. The final
        event carries the ``reply`` handed back to the user, or ``action=None`` when
        *max_steps* ran out. Call ``run`` again with the final state and the user's
        next message to continue.
    """
    state = initial_state(state_schema, state)
    schema = derive_schema(state_schema)
    for t in range(max_steps):
        prompt = render_template(
            path=PROMPT_TEMPLATE,
            instructions=instructions,
            request=request,
            state=json.dumps(state, separators=(",", ":")),
            observation=observation,
            t=t,
        )
        response = complete(
            model=model, prompt=prompt, output_schema=schema, thinking_effort="high"
        )
        assert response.output is not None
        step = response.output.model_dump()
        patch = step["patch"]
        state = update_state(state, patch)
        yield StepEvent(
            t=t,
            state=state,
            patch=patch,
            action=step["action"] or None,
            reply=step["reply"] or None,
            observation=observation,
        )
        if step["reply"]:  # turn handed back to the user
            break
        observation = yolo_bash(step["action"])
