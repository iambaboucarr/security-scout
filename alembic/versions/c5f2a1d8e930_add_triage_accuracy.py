"""add triage_accuracy

Revision ID: c5f2a1d8e930
Revises: b8c3d9e1f2a4
Create Date: 2026-04-13

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c5f2a1d8e930"
down_revision: str | None = "b8c3d9e1f2a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "triage_accuracy",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("finding_id", sa.Uuid(), nullable=False),
        sa.Column("workflow_run_id", sa.Uuid(), nullable=False),
        sa.Column(
            "predicted_ssvc_action",
            sa.Enum("immediate", "act", "attend", "track", name="ssvcaction", native_enum=False, length=32),
            nullable=True,
        ),
        sa.Column("predicted_confidence", sa.Float(), nullable=True),
        sa.Column(
            "human_decision",
            sa.Enum("approved", "rejected", "escalated", name="triagedecision", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("outcome_signal", sa.Float(), nullable=False),
        sa.Column("slack_user_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["finding_id"], ["findings.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_triage_accuracy_finding_id"), "triage_accuracy", ["finding_id"], unique=False)
    op.create_index(op.f("ix_triage_accuracy_workflow_run_id"), "triage_accuracy", ["workflow_run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_triage_accuracy_workflow_run_id"), table_name="triage_accuracy")
    op.drop_index(op.f("ix_triage_accuracy_finding_id"), table_name="triage_accuracy")
    op.drop_table("triage_accuracy")
