import os
from pathlib import Path
import uuid

import boto3
from botocore.config import Config
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def get_r2_client():
    endpoint = _require_env("R2_ENDPOINT")
    access_key = _require_env("R2_ACCESS_KEY_ID")
    secret_key = _require_env("R2_SECRET_ACCESS_KEY")
    region = os.getenv("R2_REGION", "auto")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def get_bucket() -> str:
    return _require_env("R2_BUCKET")


def get_public_base_url() -> str | None:
    base_url = os.getenv("R2_PUBLIC_BASE_URL")
    if not base_url:
        return None
    return base_url.rstrip("/")


def build_object_key(prefix: str, user_id: str, filename: str | None) -> str:
    safe_name = Path(filename or "upload.bin").name
    safe_name = safe_name.replace(" ", "_")
    unique = uuid.uuid4().hex
    return f"{prefix.rstrip('/')}/{user_id}/{unique}_{safe_name}"


def build_staged_key(user_id: str) -> str:
    unique = uuid.uuid4().hex
    return f"staged/{user_id}/{unique}.jpg"


def get_storage_url(key: str) -> str:
    base = get_public_base_url()
    if base:
        return f"{base}/{key}"
    bucket = get_bucket()
    return f"r2://{bucket}/{key}"


def create_presigned_put_url(key: str, content_type: str, expires_seconds: int = 900) -> str:
    client = get_r2_client()
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": get_bucket(), "Key": key, "ContentType": content_type},
        ExpiresIn=expires_seconds,
    )


def create_presigned_get_url(key: str, expires_seconds: int = 900) -> str:
    client = get_r2_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": get_bucket(), "Key": key},
        ExpiresIn=expires_seconds,
    )


def put_object_bytes(key: str, body: bytes, content_type: str) -> None:
    client = get_r2_client()
    client.put_object(
        Bucket=get_bucket(),
        Key=key,
        Body=body,
        ContentType=content_type,
    )
