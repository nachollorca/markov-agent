"""SKILL.state (arXiv:2608.26263) core implementation.

Whole point of the paper: prompt stays O(1) with just the necessary info,
instead of growing with the conversation.

At each step t, the model sees only three things:
    1. P: The immutable skill spec
    2. Σₜ: The current state in the given per-domain structure
    3. Oₜ: The latest observation

It returns:
    - A state update (patch) that produces Σₜ₊₁
    - An optional action that will produce Oₜ₊₁
"""

import json
import subprocess
from typing import Any, Literal

from lmdk import complete
from pydantic import BaseModel, Field, create_model

BASH_TIMEOUT = 100  # seconds to wait for the bash
MAX_BASH_OUTPUT = 100_000  # maximum chars of observation shown to the model

PROMPT = """\
{instructions}

Original request:
```
{request}
```

Current execution state (step {t}):
```json
{state}
```

Latest observation:
```
{observation}
```

Update the state, and propose the next action if necessary.

How to update the state:
- Setting a field replaces its whole current value. To add to a field, rewrite it
  in full: its current content first, then what you are adding.
- Set a field to "unchanged" to keep its current value.
- Set a field to "unset" to clear it (uninitialized fields start as "unset").
- Set the action to null when, and only when, the task is done.

Anything you will need later must be recorded in the state:
the current observation will never be shown again.\
"""


def yolo_bash(command: str) -> str:
    """Run a bash command and return it alongside its exit code, stdout and stderr."""
    # ! Probably needs a white/black- list and / or a sandbox
    proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT)
    output = (proc.stdout + proc.stderr).strip() or "(no output)"
    # The model never sees its own previous action, so the observation carries it
    return f"$ {command}\n(exit {proc.returncode})\n{output[:MAX_BASH_OUTPUT]}"


def update_state(state: dict, patch: dict) -> dict:
    """Recursively merge *patch* into *state*."""
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

    - Adds to each field the possibility to be "unchanged" or "unset"
    - Forces default value "unset" when not given
    """
    fields: dict[str, Any] = {}
    for name, field in state_schema.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            ann = derive_patch_schema(ann)
        patch_ann = ann | Literal["unchanged", "unset"]
        fields[name] = (patch_ann, Field(default="unchanged", description=field.description))
    return create_model(f"Patch_{state_schema.__name__}", **fields)


def derive_schema(state_schema: type[BaseModel]) -> type[BaseModel]:
    """Derive the step schema wrapping the state patch and action."""
    patch_schema = derive_patch_schema(state_schema)
    return create_model("Step", state=(patch_schema, ...), action=(str | None, ...))


def run(
    model: str,
    instructions: str,
    request: str,
    state_schema: type[BaseModel],
    state: dict | None = None,
    max_steps: int = 50,
) -> dict:
    """Execute a skill until the model proposes no action (done) or *max_steps* is hit.

    Args:
        model: Provider-prefixed model identifier, as in ``lmdk.complete``.
        instructions: The immutable procedural specification (P in the paper).
        state_schema: Domain-defined structure of the state (Σ in the paper) for this task.
        request: The user request, basically observation before any action (O₀).
        state: Initial execution state (Σ₀). Empty by default, can be used to start from checkpoint.
        max_steps: Horizon cap, so a looping model cannot burn tokens forever.

    Returns:
        The final execution state.
    """
    # Ensure all defaults are set
    defaults = {
        name: ("unset" if field.is_required() else field.default)
        for name, field in state_schema.model_fields.items()
    }
    state = {**defaults, **(state or {})}

    observation = "none"
    schema = derive_schema(state_schema)
    for t in range(max_steps):
        response = complete(
            model=model,
            prompt=PROMPT.format(
                instructions=instructions,
                request=request,
                state=json.dumps(state, separators=(",", ":")),
                observation=observation,
                t=t,
            ),
            output_schema=schema,
            thinking_effort="high",
        )
        assert response.output is not None
        step = response.output.model_dump()
        state = update_state(state, step["state"])
        if step["action"] is None:  # no action left to take = done
            break
        observation = yolo_bash(step["action"])
    return state
