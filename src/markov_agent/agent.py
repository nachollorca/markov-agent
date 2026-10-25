"""Domain definition for a small coding agent on top of `skill_state.run`."""

from pydantic import BaseModel, Field

DEFAULT_MODEL = "mistral:mistral-large-4"

INSTRUCTIONS = """\
You are a coding agent: you search and edit files in a project, and prove the change works.

- Look before you change: read the relevant files (ls, grep, sed -n) before editing.
- Read narrowly: grep / sed -n over whole files, the output is the observation you pay for.
- Minimal diffs: sed / python one-liners / patches; never rewrite a whole file to change one line.
- Verified means executed: run the tests or the code itself and keep its output as proof.
"""


class CodingAgentState(BaseModel):
    """State of the coding agent."""

    task: str = Field(description="The user's goal, restated in your words.")
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
