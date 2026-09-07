"""Domain definition for a small coding agent on top of `skill_state.run`."""

import json
import os

from pydantic import BaseModel, Field

from .skill_state import run

INSTRUCTIONS = """\
You are a coding agent: you edit files in a project and prove the change works.

- Look before you change: read the relevant code (ls, grep, sed -n) before editing it.
- Read narrowly: grep / sed -n over whole files, the output is the observation you pay for.
- Minimal diffs: sed / python one-liners / patches; never rewrite a whole file to change one line.
- Verified means executed: run the tests or the code itself and keep its output as proof.
"""


class CodingAgentState(BaseModel):
    """State of the coding agent."""

    task: str = Field(description="The user's goal, restated in your words.")
    working_dir: str = Field(
        description="Absolute path of the project directory. "
        "Each command runs in a fresh shell, so `cd` never persists: "
        "use absolute paths, or prefix the action commands with `cd ` and this field's value."
    )
    plan: str = Field(
        description="Remaining steps, next first. "
        "A markdown task list: one `- [ ]` item per step, `- [X]` when done.",
    )
    facts: str = Field(
        description="Everything learned from command output that you will need later: "
        "file paths, error messages, line numbers, output values. As a markdown list."
    )
    failed_approaches: str = Field(description="Commands or fixes that did not work, and why.")
    edits: str = Field(
        description="Files changed so far, and why. A markdown list, one entry per line."
    )
    verification: str = Field(
        description="Output of the run that proves the task is done.",
    )
    result: str = Field(description="One-paragraph summary for the user, written when finishing.")


if __name__ == "__main__":
    import argparse

    if os.getenv("LMDK_TELEMETRY"):
        import logfire

        logfire.configure(
            token=os.environ["LOGFIRE_TOKEN"],
            service_name="skill-state-coding-agent",
            scrubbing=False,
            send_to_logfire=True,
        )

    parser = argparse.ArgumentParser(description="Run the coding agent.")
    parser.add_argument("request", help="The user's request.")
    parser.add_argument("--model", default="vertex:gemini-3.8-flash", help="Model to use.")
    parser.add_argument("--working-dir", default=os.getcwd(), help="Project directory.")
    args = parser.parse_args()

    request, model, working_dir = args.request, args.model, args.working_dir

    with logfire.span("skill_state.run {task}", task=request):
        final_state = run(
            model=model,
            instructions=INSTRUCTIONS,
            state_schema=CodingAgentState,
            request=request,
            state={"working_dir": str(working_dir)},
        )
    print(json.dumps(final_state, indent=2))
