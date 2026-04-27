# Security Scout

A general-purpose security agent that automates vulnerability triage and validation for any repository. It monitors GitHub advisories and PRs, runs PoC validation in sandboxed containers, and routes findings through Slack approval gates before any external action.

It orchestrates existing tools (Nuclei, Semgrep, CodeQL via SARIF) and adds the missing integration layer: advisory intake, triage, validation, human approval, response.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (for Redis locally, sandbox execution in later phases)

## Quick Start

```
git clone <repo-url> && cd securityscout
cp .env.example .env
make install
docker compose up -d
make check
```

## Development

Commit and PR expectations (Conventional Commits, pre-merge checks): see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

```
make format      # auto-format with ruff
make lint        # lint (no auto-fix)
make typecheck   # mypy strict
make test        # fast tests only
make testcov     # fast tests + coverage
make testslow    # Docker/sandbox tests
make check       # all of the above (mirrors CI)
make clean       # remove caches and build artifacts
```

## Project Structure

```
src/
  agents/
  tools/
  webhooks/
tests/
sandbox/
documentation/
```

## Documentation

| Doc | Scope |
|-----|--------|
| [documentation/architecture.md](documentation/architecture.md) | Workflows, agents, state machine, tool access |
| [documentation/build-plan.md](documentation/build-plan.md) | Phased roadmap and deliverables |
| [documentation/decisions.md](documentation/decisions.md) | Architecture decisions (ADRs) |
| [documentation/tool-integrations.md](documentation/tool-integrations.md) | GitHub, Slack, webhooks, external APIs |
| [documentation/advisory-triage-polling-plan.md](documentation/advisory-triage-polling-plan.md) | **Optional** REST polling for triage/draft advisories (supplements webhooks) |
| [documentation/deployment-compose.md](documentation/deployment-compose.md) | Docker Compose deployment |
| [documentation/deployment-guide.md](documentation/deployment-guide.md) | VPS / bare-metal deployment |

## License

[Apache License 2.0](LICENSE) — see [NOTICE](NOTICE) for attribution.
