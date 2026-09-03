"""SKILL.state (arXiv:2608.26263) in ~50 lines on top of `lmdk`.

At each step t, the model sees only three things:
    1. P: The immutable skill spec
    2. Σₜ: The current state in the given per-domain structure
    3. Oₜ: The latest observation

It returns a state update (patch) and optionally an action.

Whole point of the paper: prompt stays O(1) with just the necessary info,
instead of growing with the conversation.

Deviation from the paper: the merge operator drops null-deletion semantics,
see `update_state`.
"""

import json
import subprocess

from lmdk import complete
from pydantic import BaseModel, create_model

BASH_TIMEOUT = 100  # seconds to wait for the bash
MAX_BASH_OUTPUT = 100_000  # maximum chars of observation shown to the model

PROMPT = """\
{instructions}

Current execution state:
```json
{state}
```

Latest observation:
```
{observation}
```

Update the state and propose the next action.

How to update the state:
- Setting a field replaces its whole current value. To add to a field, rewrite it
  in full: its current content first, then what you are adding.
- Leaving a field null keeps its current value. Null never erases anything, so
  fill in only the fields that actually change this step.
- Set the action to null when, and only when, the task is done.

Anything you will need later must be recorded in the state:
the current observation will never be shown again.\
"""


def yolo_bash(command: str) -> str:
    """Run a bash command and return it alongside its exit code, stdout and stderr."""
    # ! Probably needs a white/black- list and / or a sandbox
    proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT)
    output = (proc.stdout + proc.stderr).strip() or "(no output)"
    # The model never sees its own previous action, so the observation carries it.
    return f"$ {command}\n(exit {proc.returncode})\n{output[:MAX_BASH_OUTPUT]}"


def update_state(state: dict, patch: dict) -> dict:
    """Recursively merge *patch* into *state*; a null value leaves the key untouched."""
    merged = dict(state)
    for key, value in patch.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = update_state(merged[key], value)
        else:
            merged[key] = value
    return merged


def run(
    model: str,
    instructions: str,
    state_schema: type[BaseModel],
    observation: str,
    state: dict | None = None,
    max_steps: int = 50,
) -> dict:
    """Execute a skill until the model proposes no action (done) or *max_steps* is hit.

    Args:
        model: Provider-prefixed model identifier, as in ``lmdk.complete``.
        instructions: The immutable procedural specification (``P`` in the paper).
        state_schema: Domain-defined structure of the state (``Σ`` in the paper) for this task.
        observation: The user request, observation at t=0, before any action (``O₀``).
        state: Initial execution state (``Σ₀``). Empty by default.
        max_steps: Horizon cap, so a looping model cannot burn tokens forever.

    Returns:
        The final execution state.
    """
    state = state or {}
    schema = create_model("Step", state=(state_schema, ...), action=(str | None, ...))
    for _ in range(max_steps):
        response = complete(
            model=model,
            prompt=PROMPT.format(
                instructions=instructions,
                state=json.dumps(state, separators=(",", ":")),
                observation=observation,
            ),
            output_schema=schema,
            thinking_effort="high",
        )
        assert response.output is not None
        step = response.output.model_dump()  # null fields = unchanged, see `update_state`
        state = update_state(state, step["state"])
        if step["action"] is None:  # no action left to take = done
            break
        observation = yolo_bash(step["action"])
    return state
