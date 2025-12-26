import os
from typing import Tuple

import stripe


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def get_stripe() -> stripe:
    stripe.api_key = _require_env("STRIPE_SECRET_KEY")
    return stripe


def get_checkout_urls() -> Tuple[str, str]:
    success_url = _require_env("STRIPE_SUCCESS_URL")
    cancel_url = _require_env("STRIPE_CANCEL_URL")
    return success_url, cancel_url


def get_price_settings() -> Tuple[int, str]:
    price_cents = int(os.getenv("PRICE_PER_IMAGE_CENTS", "1000"))
    currency = os.getenv("PRICE_CURRENCY", "usd")
    return price_cents, currency
