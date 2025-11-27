"""Add M-Pesa transactions table

Revision ID: mpesa_001
Revises: 
Create Date: 2025-10-22

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'mpesa_001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'mpesa_transactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('checkout_request_id', sa.String(length=100), nullable=True),
        sa.Column('merchant_request_id', sa.String(length=100), nullable=True),
        sa.Column('phone_number', sa.String(length=15), nullable=True),
        sa.Column('amount', sa.Float(), nullable=True),
        sa.Column('reference', sa.String(length=100), nullable=True),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('result_code', sa.String(length=10), nullable=True),
        sa.Column('result_description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_mpesa_transactions_checkout_request_id'), 'mpesa_transactions', ['checkout_request_id'], unique=True)
    op.create_index(op.f('ix_mpesa_transactions_merchant_request_id'), 'mpesa_transactions', ['merchant_request_id'], unique=False)
    op.create_index(op.f('ix_mpesa_transactions_id'), 'mpesa_transactions', ['id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_mpesa_transactions_merchant_request_id'), table_name='mpesa_transactions')
    op.drop_index(op.f('ix_mpesa_transactions_checkout_request_id'), table_name='mpesa_transactions')
    op.drop_index(op.f('ix_mpesa_transactions_id'), table_name='mpesa_transactions')
    op.drop_table('mpesa_transactions')
