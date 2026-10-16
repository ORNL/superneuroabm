# SuperNeuroABM — working notes for Claude Code

SuperNeuroABM is the spiking-neural-network layer on top of SAGESim (a general
agent-based-modelling framework, sibling repo `../SAGESim`). Somas and synapses are
agents; a synapse delivers into its post-soma's inbox when its pre fires (event-woken),
and the synaptic kinetics live on the soma. Read `docs/EVENT_DRIVEN_NOTES.md` first: it
is the running log of the performance work (sections 15-16 are the latest
measurements and the open items), and `docs/STATUS.md` says where things stand and what
is next.

## Environment

- Python env: `/home/xxz/miniforge3/envs/sna-dev/bin/python` on the laptop (both repos
  `pip install -e`); on another machine, an env with cupy, mpi4py and both repos editable.
- One GPU is enough for multi-rank tests: `mpirun --oversubscribe -n N` with N in 1 2 3 4.
- Never run two SuperNeuroABM processes at once: setup writes the generated learning-rule
  selector to one shared file (`superneuroabm/_generated/learning_rule_selector.py`) and
  two processes with different rule sets race (`docs/API_DESIGN_NOTES.md`).
- Generated `step_func_code_*.py` files land in the working directory (gitignored);
  delete them after runs.

## Rules of the work (the owner's)

- Measure before optimizing, and prove claims by measurement or bitwise diff, not by
  assertion; re-time against NEST GPU / Brian2 after performance changes.
- Every SAGESim change must be general-purpose ABM, justified as helping any ABM.
- Discuss changes to SuperNeuroABM / SAGESim features or semantics with the owner first;
  routine fixes, tests, docs and measurements proceed.
- Never `git stash`; use `git checkout <sha> -- <path>` for a before/after state.
- Commits: short subject plus a few lines saying what was measured; the owner's other
  repos (`superneuroabm_sgnn`, `ns-applications`) are never committed by the assistant.
- Keep big plans and results in `docs/*.md` (not only in chat); API design smells found
  mid-task go into `docs/API_DESIGN_NOTES.md`.

## The gate before a commit

```sh
PY=/home/xxz/miniforge3/envs/sna-dev/bin/python
$PY -m pytest -q tests/                      # ~1 min
PYTHON=$PY bash tests/run_mpi_tests.sh       # ranks 1 2 3 4, ~5 min
# and the same two in ../SAGESim when SAGESim changed
```

Plus, for performance work, the Brunel probe in `scaling_analysis/per_tick_probes/`
(same spikes at every rank count) and the numbers in `docs/EVENT_DRIVEN_NOTES.md`.
