# CHANGELOG

<!-- version list -->

## v1.4.0 (2026-10-25)

### Bug Fixes

- **core**: Make sure agent fills only action or reply
  ([`1970bee`](https://github.com/nachollorca/markov-agent/commit/1970beeb7a593e19c90d1fe26c0e30bec1f7e4b3))

- **core**: Remove working dir, since it is not necessary
  ([`052020d`](https://github.com/nachollorca/markov-agent/commit/052020d02b1e9fd0dae5e800c51c8d08cafa1d85))

- **core**: Separate the last action from the last observation
  ([`fdd7473`](https://github.com/nachollorca/markov-agent/commit/fdd74733cfe090c323f6063793abe6303c080813))

- **prompt**: Make reply instructions clearer
  ([`824a49f`](https://github.com/nachollorca/markov-agent/commit/824a49f757b5aa7b3d837a54d87c4fb3480c3532))

- **prompt**: Remove instruction repetition, move each thing to its place
  ([`fe788dd`](https://github.com/nachollorca/markov-agent/commit/fe788dd65b819726d092914f9a6c50f391377b32))

- **tui**: Add margins
  ([`f05aa0f`](https://github.com/nachollorca/markov-agent/commit/f05aa0fcc012312be5027c32c1fc919239399e23))

### Documentation

- **readme**: Add figure
  ([`a4039c0`](https://github.com/nachollorca/markov-agent/commit/a4039c0f1d6a05897882b6f0279bc56cbd9e21bd))

- **readme**: Link the harness design paper
  ([`49d4703`](https://github.com/nachollorca/markov-agent/commit/49d4703b70b569d92c5b5c7871d30d3aab28eb61))

- **readme**: Remove placeholder
  ([`37ac3ad`](https://github.com/nachollorca/markov-agent/commit/37ac3ad6dc04d0e29bee1b0c483888968ff8ac88))

### Features

- **multi-turn**: User becomes environment, agent can act on it
  ([`5ee548c`](https://github.com/nachollorca/markov-agent/commit/5ee548c594baec74c2bcf255cef6c67cf1f27243))

- **tui**: Add skeleton
  ([`ff56bbc`](https://github.com/nachollorca/markov-agent/commit/ff56bbcdac6d2993f9e494d9fccb1c7beadc7bad))

- **tui**: Initial implementation
  ([`f38e885`](https://github.com/nachollorca/markov-agent/commit/f38e885f6d58d24aaf6d1861d1511207e529bb9c))

### Refactoring

- **tui**: Delete the initial REPL, now will live in TUI
  ([`f675025`](https://github.com/nachollorca/markov-agent/commit/f675025b3901b10ca6b9a713bc5bdcabe7b54725))


## v1.3.0 (2026-09-19)

### Bug Fixes

- **demo**: Use mistral instead of gemini
  ([`8075227`](https://github.com/nachollorca/markov-agent/commit/80752276d3b50da746e1b3058ccc76fae5949c46))

### Code Style

- Little changes to wording
  ([`baca469`](https://github.com/nachollorca/markov-agent/commit/baca46961c55289fd1214a2c5a95c810f9f2b547))

### Documentation

- **readme**: Change wording
  ([`10f6e81`](https://github.com/nachollorca/markov-agent/commit/10f6e81af03dc06f89287313c8be3df48d73a5be))

### Features

- Make StepEvent
  ([`627cee5`](https://github.com/nachollorca/markov-agent/commit/627cee54c71238f10e670d77636c03e464469f73))

- **core**: Make run emit step events, make proper use of `state` and `patch` when naming symbols
  ([`9928008`](https://github.com/nachollorca/markov-agent/commit/99280089ab53972772ea56b14e550c9b85cc8df2))

### Refactoring

- **core**: Make a helper to initialize the fields
  ([`1501354`](https://github.com/nachollorca/markov-agent/commit/15013544ee4623fb9a93b7be000983cd41b03c6e))

- **prompt**: Use jinja instead of hard coded string
  ([`44774ee`](https://github.com/nachollorca/markov-agent/commit/44774eef66b78f3b916b6a5543d6e2cba2c8ba07))


## v1.2.0 (2026-09-07)

### Chores

- Move logfire to dev deps
  ([`87a6a85`](https://github.com/nachollorca/markov-agent/commit/87a6a85f22e990ee868159e4b207943c4f7e5d20))

- Rename project to markov-agent
  ([`ba33845`](https://github.com/nachollorca/markov-agent/commit/ba33845871fd0b5dcafca37e98a09d0dd7574924))

### Documentation

- **readme**: Explain why markov
  ([`ebe1de2`](https://github.com/nachollorca/markov-agent/commit/ebe1de2add155578f2acc6f536b73bbb54c3b324))

- **readme**: Make the markov interpretation shorter
  ([`c830869`](https://github.com/nachollorca/markov-agent/commit/c830869c3902ed50482e7fc71673e47701b37b80))

### Features

- **prompt**: Make functioning instructions clearer
  ([`d06aa87`](https://github.com/nachollorca/markov-agent/commit/d06aa87ae6e16bf8618c6d061d7fe08c6e123639))


## v1.1.0 (2026-09-04)

### Features

- Add more meat to the primpt
  ([`b8a3082`](https://github.com/nachollorca/markov-agent/commit/b8a3082ec19f87a8c63ae48d10f459f343b7a01d))

- Improve the patching mechanism
  ([`76f178a`](https://github.com/nachollorca/markov-agent/commit/76f178a0847f83c245cdab3e4bc06013b9db67ac))

### Testing

- Increase coverage
  ([`ad0c22f`](https://github.com/nachollorca/markov-agent/commit/ad0c22f5db0ffcda8cd9df9d2580c0b3cb7d171a))


## v1.0.0 (2026-09-03)

- Initial Release
