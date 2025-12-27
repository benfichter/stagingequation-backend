import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_recipients() -> list[str]:
    raw = os.getenv("ORDER_NOTIFY_EMAILS", "")
    return [email.strip() for email in raw.split(",") if email.strip()]


def send_order_notification(
    *,
    order_id: str,
    user_id: str,
    user_name: str,
    user_email: str,
    user_firm: str,
    image_count: int,
    amount_cents: int,
    currency: str,
    note: str | None,
) -> None:
    recipients = _get_recipients()
    if not recipients:
        logger.info("Order notification skipped: ORDER_NOTIFY_EMAILS not set")
        return

    host = os.getenv("SMTP_HOST")
    if not host:
        logger.info("Order notification skipped: SMTP_HOST not set")
        return

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM", username or "")
    if not from_email:
        logger.info("Order notification skipped: SMTP_FROM not set")
        return

    use_tls = _parse_bool(os.getenv("SMTP_USE_TLS", "true"), default=True)
    use_ssl = _parse_bool(os.getenv("SMTP_USE_SSL", "false"))

    total = amount_cents / 100
    subject = f"New staging order {order_id}"
    body = "\n".join(
        [
            "A new staging order was placed.",
            "",
            f"Order ID: {order_id}",
            f"User ID: {user_id}",
            f"Firm: {user_firm}",
            f"Name: {user_name}",
            f"Email: {user_email}",
            f"Images: {image_count}",
            f"Total: {total:.2f} {currency.upper()}",
            f"Note: {note or '(none)'}",
        ]
    )

    message = EmailMessage()
    message["From"] = from_email
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body)

    try:
        if use_ssl:
            server: smtplib.SMTP = smtplib.SMTP_SSL(host, port)
        else:
            server = smtplib.SMTP(host, port)
        with server:
            if use_tls and not use_ssl:
                server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(message)
        logger.info(
            "Order notification sent for order %s to %s recipient(s)",
            order_id,
            len(recipients),
        )
    except Exception:
        logger.exception("Order notification failed")
