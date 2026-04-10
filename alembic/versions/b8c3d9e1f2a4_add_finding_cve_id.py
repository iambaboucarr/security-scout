"""add finding cve_id

Revision ID: b8c3d9e1f2a4
Revises: a41027ef13ac
Create Date: 2026-04-10

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b8c3d9e1f2a4"
down_revision: str | None = "a41027ef13ac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("findings", sa.Column("cve_id", sa.String(length=64), nullable=True))
    op.create_index(op.f("ix_findings_cve_id"), "findings", ["cve_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_findings_cve_id"), table_name="findings")
    op.drop_column("findings", "cve_id")
