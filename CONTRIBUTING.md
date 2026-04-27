# Contributing

Thank you for helping with Security Scout. This document covers **licensing**, **commit messages**, and **pre-merge checks**. For architecture and coding rules, see `CLAUDE.md` and `docs/`.

## Licensing your contribution

Security Scout is licensed under the [Apache License 2.0](LICENSE). By submitting a pull request or patch, you agree that your contribution is licensed under the same terms (Apache License 2.0, Section 5 — inbound = outbound). No separate Contributor License Agreement (CLA) is required.

If your employer has intellectual property policies, please ensure you have permission to contribute before submitting.

We recommend (but do not yet require) signing off your commits with the [Developer Certificate of Origin](https://developercertificate.org/) (`git commit -s`) to certify you have the right to submit the contribution.

## Commit messages (Conventional Commits)

We follow **[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)**: a small grammar for commit messages so history and tooling stay readable. The spec was inspired by the **[Angular](https://github.com/angular/angular)** team’s commit conventions; Angular still maintains an extended guide ([commit message guidelines](https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md)) that many teams use as a reference for types, scopes, and bodies.

### Format

```
<type>(<scope>): <short summary>

Optional body: explain *why*, reference issues, note follow-ups.
Use a blank line between subject and body.

Optional footer: BREAKING CHANGE: … or Fixes #123
```

- **Subject line**: imperative mood (“Add…”, “Fix…”), not past tense. Keep the summary short (often ≤ 50–72 characters). No trailing period on the subject is common practice (see Angular’s guide).
- **Type** (common): `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`, `build`.
- **Scope** (optional but encouraged): area of the codebase, e.g. `tools`, `webhooks`, `agents`, `config`, `worker`, `db`, `ci`, `deps`.
- **Breaking changes**: append `!` after the type or scope (e.g. `feat(api)!: …`) and/or a `BREAKING CHANGE:` paragraph in the footer, per the spec.

### Examples

```
feat(tools): add GitHub advisory fetch helper

docs(tools): tighten module docstrings

fix(webhooks): reject replayed payloads outside window

chore(ci): pin action digests for zizmor
```

## Before you open a PR

- Run **`make check`** (lint, typecheck, SQLite test suite, then the `@pytest.mark.postgres` suite). For the Postgres suite locally, run **`docker compose up -d postgres`** (or set **`POSTGRES_TEST_URL`**); **`make test`** stays SQLite-only for a fast loop.
- Keep commits **atomic** when practical: one logical change per commit.
- Do not commit secrets; `.env` stays local (see `.env.example`).

## Tests (pytest vs Make)

- **`make test`** passes **`-m "not postgres"`**, so you do not need Postgres for the default loop.
- **`uv run pytest`** with **no `-m`** collects **every** test, including **`@pytest.mark.postgres`**, which **`pytest.fail`**s if **`POSTGRES_TEST_URL`** is unset. For IDE “run all tests” without Postgres, use **`-m "not postgres"`** (or set **`POSTGRES_TEST_URL`**).
- Pytest **AND**s multiple **`-m`** expressions; the Makefile uses explicit **`-m`** flags instead of putting **`not postgres`** in **`addopts`** so **`pytest -m postgres`** still works.
- With **pytest-xdist** (**`-n auto`**) and **`@pytest.mark.postgres`**, use **`--dist loadgroup`** so tests that share one database schema do not run DDL in parallel (CI and **`Makefile`** targets include this).
- **`make testslow`** / **`make testintegration`** do not exclude **`postgres`**. If a test is marked **`slow`** or **`integration`** and **`postgres`**, run **`make services`** (or set **`POSTGRES_TEST_URL`**) before that target.

## Further reading

| Resource                                                                                                                           | Role                                                                   |
| ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)                                                       | Normative commit grammar                                               |
| [Angular — commit message guidelines](https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md) | Detailed examples and team conventions (historical basis for the spec) |
| `CLAUDE.md`                                                                                                                        | Project-specific implementation standards                              |
| `.cursor/skills/fastapi/SKILL.md`                                                                                                 | FastAPI / Pydantic agent guidance (see `references/security-scout.md` for repo overrides) |
