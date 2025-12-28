"""Add password hash to users.

Revision ID: 0004_add_user_password_hash
Revises: 0003_orders
Create Date: 2025-12-28 19:05:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0004_add_user_password_hash"
down_revision = "0003_orders"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "password_hash")
