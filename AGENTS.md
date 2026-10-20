# Working in this repo

- Branch model: `dev` is the default and deploy branch; feature branches (`feat/…`, `fix/…`, `chore/…`, Conventional Commits) branch off `dev` and PR back into `dev`. `main` is reserved for prod and does not exist yet. The org ruleset requires a PR and the **security / Security Gate** check.
- Secrets: never in the repo or in GitHub Actions. This repo's Infisical project is **`bacmr-py-common`** (sops.bacmr.com); deploys render env files on the VPS from it via `deploy/refresh-env.sh` validated against `deploy/env/*.template` (to be added with the first deployable change).
- Docs: this README is the contract. Architecture decisions live in `BacMR/docs` (ADRs); the roadmap is the **Production Road Map** project board.
- Status and provenance are in the README's *Status* section; keep it current in the same PR as the change.
- Versioning: this is a library, no Infisical project is used and nothing deploys. Release = tag `vX.Y.Z` on `dev` (bump `version` in `pyproject.toml` in the same PR) **before** any dependent repo bumps its pin. Never retag.
- CI (`ci.yml`): `uv sync --locked`, `ruff check`, `pytest`. Commit `uv.lock` with dependency changes.
