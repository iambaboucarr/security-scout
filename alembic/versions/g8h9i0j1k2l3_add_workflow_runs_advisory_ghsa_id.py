"""add workflow_runs.advisory_ghsa_id

Revision ID: g8h9i0j1k2l3
Revises: f1e2d3c4b5a6
Create Date: 2026-04-27

"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "g8h9i0j1k2l3"
down_revision: str | None = "f1e2d3c4b5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_log = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    op.add_column(
        "workflow_runs",
        sa.Column("advisory_ghsa_id", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_workflow_runs_advisory_ghsa_id",
        "workflow_runs",
        ["advisory_ghsa_id"],
    )

    bind = op.get_bind()
    d = bind.dialect.name
    if d == "postgresql":
        op.execute(
            sa.text(
                """
                UPDATE workflow_runs wr
                SET advisory_ghsa_id = UPPER(TRIM(f.evidence->>'ghsa_id'))
                FROM findings f
                WHERE f.id = wr.finding_id
                  AND (wr.advisory_ghsa_id IS NULL)
                  AND f.evidence->>'ghsa_id' IS NOT NULL
                """
            )
        )
    elif d == "sqlite":
        op.execute(
            sa.text(
                """
                UPDATE workflow_runs
                SET advisory_ghsa_id = UPPER(TRIM((
                    SELECT json_extract(f.evidence, '$.ghsa_id')
                    FROM findings f
                    WHERE f.id = workflow_runs.finding_id
                )))
                WHERE finding_id IS NOT NULL
                  AND (advisory_ghsa_id IS NULL)
                """
            )
        )
    else:
        _log.info(
            "Skipping workflow_runs.advisory_ghsa_id backfill: dialect is %s",
            d,
        )


def downgrade() -> None:
    op.drop_index("ix_workflow_runs_advisory_ghsa_id", table_name="workflow_runs")
    op.drop_column("workflow_runs", "advisory_ghsa_id")
