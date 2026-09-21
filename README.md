# markov-agent

A coding agent following the principles defined in the [SKILL.state](docs/paper.md) paper:

> Runtime architecture that replaces append-only conversational history with an explicit, mutable execution state. At each execution step, the model receives only the immutable skill specification, the current structured execution state, and the latest observation. Intermediate reasoning is discarded immediately after **producing a validated state update**, preventing prompt growth with execution history.

Hence, the context given to the agent is modeled similar to a Markov process: the next state depends only on the current one and the (observed result of the) action taken.

## Overview

Although the paper specifies tasks like [...complete...], I wanted to use it for a coding agent and test it in tasks I normally through at [Pi](https://github.com/earendil-works/pi) and see what happens.

The action space is: **bash**. Nothing more and nothing less. While most coding agents have some specially guardrailed tools (`read`, `patch`, etc.), [there is research proving that powerful LMs can do better without](https://arxiv.org/pdf/2609.20804):

> Predefined tools improve performance for models with weaker bash proficiency, whereas bash-capable models can operate effectively with a bash-only interface and achieve substantially lower cost, especially on command-line-centric tasks.

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

## References

```
@misc{badhe2026skillstatescalablelonghorizonagent,
      title={SKILL.state: Scalable Long-Horizon Agent Skills},
      author={Sanket Badhe and Priyanka Tiwari and Jonghyun Chung},
      year={2026},
      eprint={2608.26263},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.26263},
}
```
