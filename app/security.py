from passlib.context import CryptContext

_BCRYPT_MAX_BYTES = 72

_PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _ensure_password_length(password: str) -> None:
    if len(password.encode("utf-8")) > _BCRYPT_MAX_BYTES:
        raise ValueError("Password must be 72 bytes or fewer.")


def hash_password(password: str) -> str:
    _ensure_password_length(password)
    return _PWD_CONTEXT.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    _ensure_password_length(password)
    return _PWD_CONTEXT.verify(password, password_hash)
