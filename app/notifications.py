import html
import logging
import os
from typing import Iterable

import resend

from app import models

logger = logging.getLogger(__name__)

DEFAULT_ADMIN_EMAILS = [
    "eshaam.bhattad@gmail.com",
    "haha53raid@gmail.com",
    "stealthpolymath@gmail.com",
]


def _get_admin_recipients() -> list[str]:
    configured = os.getenv("ADMIN_NOTIFICATION_EMAILS", "")
    if configured:
        return [email.strip() for email in configured.split(",") if email.strip()]
    return DEFAULT_ADMIN_EMAILS


def _format_currency(amount_cents: int, currency: str) -> str:
    return f"${amount_cents / 100:.2f} {currency.upper()}"


def _build_upload_list(uploads: Iterable[models.Upload]) -> str:
    items = []
    for upload in uploads:
        name = html.escape(upload.original_filename or "image")
        url = html.escape(upload.storage_url or "")
        if url:
            items.append(f"<li><strong>{name}</strong><br /><a href=\"{url}\">{url}</a></li>")
        else:
            items.append(f"<li><strong>{name}</strong></li>")
    if not items:
        return "<li>No uploads attached.</li>"
    return "\n".join(items)


def send_order_created_notification(
    order: models.Order,
    user: models.User,
    uploads: list[models.Upload],
) -> bool:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.warning("RESEND_API_KEY not configured; skipping order notification email.")
        return False

    resend.api_key = api_key
    from_address = os.getenv("RESEND_FROM", "hello@resend.com")
    recipients = _get_admin_recipients()
    if not recipients:
        logger.warning("No admin notification recipients configured.")
        return False

    safe_note = html.escape(order.note or "None")
    safe_name = html.escape(user.name)
    safe_email = html.escape(user.email)
    safe_firm = html.escape(user.firm_name)
    safe_phone = html.escape(user.phone or "Not provided")
    created_at = order.created_at.isoformat() if order.created_at else "Unknown"
    total_display = _format_currency(order.amount_cents, order.currency)
    uploads_html = _build_upload_list(uploads)

    params = {
        "from": from_address,
        "to": recipients,
        "subject": f"New Order Created - {order.id}",
        "html": f"""
        <h2>New Order Created</h2>
        <p><strong>Order ID:</strong> {order.id}</p>
        <p><strong>Status:</strong> {html.escape(order.status)}</p>
        <p><strong>Created:</strong> {created_at}</p>
        <p><strong>Images:</strong> {order.image_count}</p>
        <p><strong>Total:</strong> {total_display}</p>
        <h3>Customer</h3>
        <p><strong>Name:</strong> {safe_name}</p>
        <p><strong>Email:</strong> {safe_email}</p>
        <p><strong>Firm:</strong> {safe_firm}</p>
        <p><strong>Phone:</strong> {safe_phone}</p>
        <h3>Note</h3>
        <p>{safe_note}</p>
        <h3>Uploads</h3>
        <ul>
            {uploads_html}
        </ul>
        <hr />
        <p><small>Delivery target: 24 hours, via email.</small></p>
        """,
    }

    try:
        resend.Emails.send(params)
        logger.info("Order notification sent to admin recipients.")
        return True
    except Exception as exc:
        logger.exception("Email send error: %s", exc)
        return False
