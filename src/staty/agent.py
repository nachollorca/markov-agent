"""Domain definition for a small coding agent on top of `skill_state.run`."""

import json
import os

from pydantic import BaseModel, Field

from .skill_state import run

INSTRUCTIONS = """\
You are a coding agent working in a Linux shell.

Tips and tricks:
- Look before you change: read the relevant files (sed -n, grep, ls) before editing.
- One bash command per step; prefer precise commands (grep, sed) over dumping whole files.
- After editing, verify by running the change before declaring the task done.
- Minimal diffs: sed / python one-liners / patches; never rewrite a whole file to change one line.
- Never re-run a command whose output is already in `facts`.
- Check `failed_approaches` before retrying; append there when a command doesn't do what expected.
"""


class CodingAgentState(BaseModel):
    """State of the coding agent."""

    task: str | None = Field(default=None, description="The user's goal, restated in your words.")
    working_dir: str | None = Field(
        default=None,
        description="Absolute path of the project directory. The shell does not persist `cd`: "
        "every command must start with `cd {working_dir} && ...` or use absolute paths.",
    )
    plan: str | None = Field(
        default=None,
        description="Remaining steps, next first. A markdown task list: one `- [ ]` item per step, "
        "`- [X]` when done.",
    )
    facts: str | None = Field(
        default=None,
        description="Everything learned from command output that you will need later: "
        "file paths, error messages, line numbers, output values. As a markdown list.",
    )
    failed_approaches: str | None = Field(
        default=None,
        description="Commands or fixes that did not work, and why. Never retry these. "
        "A markdown list, one entry per line.",
    )
    edits: str | None = Field(
        default=None,
        description="Files changed so far, and why. A markdown list, one entry per line.",
    )
    verification: str | None = Field(
        default=None,
        description="Output of the run that proves the task is done: actual evidence, not a claim.",
    )
    result: str | None = Field(
        default=None, description="One-paragraph summary for the user, written when finishing."
    )


if __name__ == "__main__":
    import argparse

    import logfire

    parser = argparse.ArgumentParser(description="Run the coding agent.")
    parser.add_argument("request", help="The user's request.")
    parser.add_argument("--model", default="vertex:gemini-3.8-flash", help="Model to use.")
    parser.add_argument("--working-dir", default=os.getcwd(), help="Project directory.")
    args = parser.parse_args()

    request, model, working_dir = args.request, args.model, args.working_dir

    logfire.configure(
        token=os.environ["LOGFIRE_TOKEN"],
        service_name="skill-state-coding-agent",
        scrubbing=False,
        send_to_logfire=True,
    )
    with logfire.span("skill_state.run {task}", task=request):
        final_state = run(
            model=model,
            instructions=INSTRUCTIONS,
            state_schema=CodingAgentState,
            observation=request,
            state={"working_dir": str(working_dir)},
        )
    print(json.dumps(final_state, indent=2))
