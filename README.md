# markov-agent

A coding agent following [SKILL.state](docs/paper.md) paper principles.

The context given to the agent is modeled as a Markov process: the next state depends only on the current one and the action taken — not on the full history. Such state is updated at every step via an explicit `transition(state, action) -> new_state` function. Therefore, the prompt stays `O(1)` with just the necessary info instead of growing with the conversation.

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
