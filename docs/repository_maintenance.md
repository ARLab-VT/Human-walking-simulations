# Repository maintenance

This document defines how to maintain the ARLab-VT human-walking simulation
study while preserving reproducibility and its relationship to upstream
MuscleMimic.

## Repository roles

Use two Git remotes:

| Remote | Repository | Role |
| --- | --- | --- |
| `origin` | `git@github.com:ARLab-VT/Human-walking-simulations.git` | ARLab-VT study development, reviews, and releases. |
| `upstream` | `https://github.com/amathislab/musclemimic.git` | Read-only source for framework updates and attribution. |

The protected `main` branch should contain reviewed, reproducible work. Develop
the reflex-recovery study on `research/reflex-recovery` or on smaller branches
created from it. Merge through pull requests after validation.

## Standard development cycle

```bash
git switch research/reflex-recovery
git pull --ff-only origin research/reflex-recovery
git switch -c feature/<short-description>

# edit and validate
git status --short
git diff --check
uv run pytest -q tests/reflex_recovery tests/test_checkpoint_canonicalization.py

git add <explicit-paths>
git commit -m "<area>: <concise change>"
git push -u origin feature/<short-description>
```

Open a pull request into `research/reflex-recovery` for experimental changes.
Promote a validated study snapshot from `research/reflex-recovery` into `main`
with a separate pull request.

## Upstream synchronization

Fetch upstream without changing the study branch:

```bash
git fetch upstream
git log --oneline --left-right research/reflex-recovery...upstream/main
git switch -c maintenance/sync-upstream research/reflex-recovery
git merge upstream/main
```

Resolve conflicts deliberately, rerun the full focused validation, and use a
pull request to merge the synchronization branch. Re-run the exact-model audit,
baseline comparison, and notebook whenever upstream changes the environment,
model, action path, checkpoint loader, or evaluation code.

## Required validation

Before merging implementation changes:

1. Run `git diff --check`.
2. Run the focused reflex-recovery tests and checkpoint canonicalization test.
3. Validate the notebook JSON and every relative Markdown link.
4. Run the 4,096-step integration smoke gate for environment/controller changes.
5. Reproduce matched rollouts for changes affecting dynamics, actions, rewards,
   termination, checkpoints, or evaluation.
6. Record the configuration, seed, source commit, checkpoint identity, dataset
   identity, backend, and dependency versions for scientific evidence.

## Artifact policy

Commit source, configurations, tests, lightweight documentation, summary tables,
and compact figures needed for review. Do not commit:

- Hugging Face datasets or gated motion data;
- policy checkpoints or optimizer state;
- JAX compilation caches;
- raw rollout arrays;
- generated videos or large rendered media;
- access tokens, SSH keys, credentials, or machine-specific cache paths.

These paths and file types are covered by `.gitignore`. Store reproducible large
artifacts in a versioned GitHub Release, institutional storage, or an approved
Hugging Face repository, and record their immutable identifier and checksum in
the release notes.

## Release checklist

1. Confirm the notebook's phase-status table matches the actual evidence.
2. Run focused tests and the required simulation gates.
3. Generate matched plots, metrics, and videos from a clean checkout.
4. Confirm no credentials, gated data, checkpoints, caches, or private paths are
   tracked.
5. Tag the reviewed commit with a study version such as `reflex-recovery-v0.1.0`.
6. Publish checksums and provenance for external artifacts.
7. Document limitations and open scientific gates in the release notes.
