from dataclasses import dataclass

from pydantic import BaseModel

from markov_agent.skill_state import (
    MAX_BASH_OUTPUT,
    derive_patch_schema,
    derive_schema,
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
    out = yolo_bash(f"python3 -c 'print(\"$\" * {MAX_BASH_OUTPUT + 10})'")
    assert out.endswith("\n" + "$" * MAX_BASH_OUTPUT)


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


def test_derive_schema_wraps_patch_and_action():
    class S(BaseModel):
        x: int

    step_schema = derive_schema(S)
    assert "state" in step_schema.model_fields
    assert "action" in step_schema.model_fields


def test_non_dict_patch_replaces_dict_value():
    assert update_state({"a": {"b": 1}}, {"a": 2}) == {"a": 2}


def test_dict_patch_replaces_scalar_value():
    assert update_state({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}


def test_run_executes_actions_until_done(monkeypatch):
    import markov_agent.skill_state as skill_state

    class FakeState(BaseModel):
        n: int

    @dataclass(frozen=True)
    class FakeStep(BaseModel):
        state: dict
        action: str | None

    @dataclass(frozen=True)
    class FakeResponse:
        output: FakeStep

    steps = iter(
        [FakeStep(state={"n": 1}, action="echo one"), FakeStep(state={"n": 2}, action=None)]
    )
    calls = []

    def fake_complete(**kwargs):
        calls.append(kwargs["prompt"])
        return FakeResponse(next(steps))

    monkeypatch.setattr(skill_state, "complete", fake_complete)
    assert skill_state.run("m", "i", "r", FakeState) == {"n": 2}
    assert len(calls) == 2 and "(exit 0)\none" in calls[1] and "step 0" in calls[0]
    # initial state seeded 'n' with 'unset'
    assert '"n":"unset"' in calls[0]
