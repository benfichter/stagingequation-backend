"""Add upload kind and parent reference.

Revision ID: 0002_uploads_kind_parent
Revises: 0001_init
Create Date: 2025-12-26 12:30:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_uploads_kind_parent"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "uploads",
        sa.Column("kind", sa.String(length=50), server_default="original", nullable=False),
    )
    op.add_column(
        "uploads",
        sa.Column("parent_upload_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_uploads_parent_upload_id",
        "uploads",
        "uploads",
        ["parent_upload_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_uploads_parent_upload_id", "uploads", ["parent_upload_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_uploads_parent_upload_id", table_name="uploads")
    op.drop_constraint("fk_uploads_parent_upload_id", "uploads", type_="foreignkey")
    op.drop_column("uploads", "parent_upload_id")
    op.drop_column("uploads", "kind")
