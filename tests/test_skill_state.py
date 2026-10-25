from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ValidationError

from markov_agent.skill_state import (
    MAX_BASH_OUTPUT,
    derive_patch_schema,
    derive_schema,
    initial_state,
    update_state,
    yolo_bash,
)


def test_unchanged_keeps_the_current_value():
    assert update_state({"task": "t", "plan": "p"}, {"plan": "unchanged"}) == {
        "task": "t",
        "plan": "p",
    }


def test_unset_sets_value_to_unset():
    assert update_state({"task": "t", "plan": "p"}, {"plan": "unset"}) == {
        "task": "t",
        "plan": "unset",
    }


def test_set_field_replaces_and_others_survive():
    assert update_state({"task": "t", "plan": "p"}, {"plan": "q"}) == {"task": "t", "plan": "q"}


def test_nested_dicts_merge_key_by_key():
    state = {"files": {"a.py": "todo", "b.py": "done"}}
    assert update_state(state, {"files": {"a.py": "done"}}) == {
        "files": {"a.py": "done", "b.py": "done"}
    }


def test_observation_carries_the_command_and_exit_code():
    assert yolo_bash("echo hi") == "$ echo hi\n(exit 0)\nhi"
    assert yolo_bash("exit 3") == "$ exit 3\n(exit 3)\n(no output)"


def test_output_is_truncated_to_max_chars():
    command = f"python3 -c 'print(\"$\" * {MAX_BASH_OUTPUT + 10})'"
    out = yolo_bash(command)
    assert out == (
        f"$ {command}\n(exit 0)\n"
        + "$" * MAX_BASH_OUTPUT
        + f"\n[The output has been truncated at the maximum {MAX_BASH_OUTPUT} characters]"
    )


def test_patch_cannot_mutate_input_state():
    state = {"a": {"b": 1}}
    update_state(state, {"a": {"c": 2}})
    assert state == {"a": {"b": 1}}


def test_derive_patch_schema_adds_unchanged_and_unset():
    class S(BaseModel):
        x: int
        y: str

    Patch = derive_patch_schema(S)
    schema = Patch.model_json_schema()
    for prop in schema["properties"].values():
        enum_types = [opt.get("enum") for opt in prop.get("anyOf", []) if "enum" in opt]
        assert ["unchanged", "unset"] in enum_types


def test_initial_state_seeds_unset_and_defaults():
    class S(BaseModel):
        required: int
        optional: str = "d"

    assert initial_state(S) == {"required": "unset", "optional": "d"}
    assert initial_state(S, {"required": 1}) == {"required": 1, "optional": "d"}


def test_derive_schema_wraps_patch_and_action():
    class S(BaseModel):
        x: int

    step_schema = derive_schema(S)
    assert "patch" in step_schema.model_fields
    assert "action" in step_schema.model_fields
    assert "reply" in step_schema.model_fields


def test_step_needs_exactly_one_of_action_and_reply():
    class S(BaseModel):
        x: int

    step_schema = derive_schema(S)
    assert (
        step_schema.model_validate({"patch": {}, "reply": "which file?"}).model_dump()["action"]
        == ""
    )
    # the strict provider schema marks both required, so "unchanged" lands in the empty one
    assert step_schema.model_validate({"patch": {}, "action": "ls", "reply": "unchanged"})
    with pytest.raises(ValidationError, match="exactly one"):
        step_schema.model_validate({"patch": {}, "action": "ls", "reply": "done"})
    with pytest.raises(ValidationError, match="exactly one"):
        step_schema.model_validate({"patch": {}})


def test_non_dict_patch_replaces_dict_value():
    assert update_state({"a": {"b": 1}}, {"a": 2}) == {"a": 2}


def test_dict_patch_replaces_scalar_value():
    assert update_state({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}


@dataclass
class FakeStep:
    patch: dict
    action: str | None = None
    reply: str | None = None

    def model_dump(self):
        return {"patch": self.patch, "action": self.action, "reply": self.reply}


def _fake_run(monkeypatch, steps):
    """Wire run() to a canned list of steps; returns the module, state class and prompts."""
    import markov_agent.skill_state as skill_state

    class FakeState(BaseModel):
        n: int

    remaining = iter(steps)
    calls = []

    def fake_complete(**kwargs):
        calls.append(kwargs["prompt"])
        return type("R", (), {"output": next(remaining)})()

    monkeypatch.setattr(skill_state, "complete", fake_complete)
    return skill_state, FakeState, calls


def test_run_executes_actions_until_the_agent_replies(monkeypatch):
    skill_state, FakeState, calls = _fake_run(
        monkeypatch,
        [
            FakeStep(patch={"n": 1}, action="echo one"),
            FakeStep(patch={"n": 2}, reply="one"),
            FakeStep(patch={"n": 3}, action="never runs"),  # the reply ended the turn
        ],
    )
    events = list(skill_state.run("m", "i", "r", FakeState))
    assert [e.state for e in events] == [{"n": 1}, {"n": 2}]
    assert events[-1].action is None
    assert events[-1].reply == "one"
    assert events[0].observation == "none"
    assert events[1].observation == "$ echo one\n(exit 0)\none"
    assert len(calls) == 2 and "(exit 0)\none" in calls[1] and "step 0" in calls[0]
    # initial state seeded 'n' with 'unset'
    assert '"n":"unset"' in calls[0]


def test_the_user_reply_resumes_the_turn_as_the_next_observation(monkeypatch):
    skill_state, FakeState, calls = _fake_run(
        monkeypatch,
        [FakeStep(patch={"n": 1}, reply="which file?"), FakeStep(patch={"n": 2}, reply="done")],
    )
    first = list(skill_state.run("m", "i", "r", FakeState))
    second = list(skill_state.run("m", "i", "r", FakeState, first[-1].state, "a.py"))
    assert first[-1].reply == "which file?"
    assert [e.state for e in second] == [{"n": 2}]
    assert second[0].observation == "a.py"  # the user is part of the environment
    assert "Latest observation:\n```\na.py\n```" in calls[1]
    assert "Request:\n```\nr\n```" in calls[1]  # the request stays the session's task
