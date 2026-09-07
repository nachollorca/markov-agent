# markov-agent

A coding agent following the principles defined in the [SKILL.state](docs/paper.md) paper:

> Runtime architecture that replaces append-only conversational history with an explicit, mutable execution state. At each execution step, the model receives only the immutable skill specification, the current structured execution state, and the latest observation. Intermediate reasoning is discarded immediately after **producing a validated state update**, preventing prompt growth with execution history.

The context given to the agent is thus modeled similar to a Markov process: the next state depends only on the current one and the (observed result of the) action taken.

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
