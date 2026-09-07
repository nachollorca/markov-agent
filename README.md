# markov-agent

A coding agent following [SKILL.state](docs/paper.md) paper principles.

The name comes from the paper's core idea: agent state is modeled as a Markov process, where the next state depends only on the current state and the action taken — not on the full history. The agent's context (state) is updated via an explicit `transition(state, action) -> new_state` function, making state evolution predictable and testable rather than an implicit side effect of the conversation.

## Installation

```bash
uv add markov-agent
```

## Usage

```python
import markov_agent
```

## Development

This project uses [uv](https://docs.astral.sh/uv/), [just](https://just.systems/) and [prek](https://github.com/j178/prek).

See [justfile](justfile) for pre-built commands used along pre-commit hooks and CI.

_Made with [`mold`](https://github.com/nachollorca/mold) template_
