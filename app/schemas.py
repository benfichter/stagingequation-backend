from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field
from pydantic import ConfigDict


class UserCreate(BaseModel):
    firm_name: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    phone: Optional[str] = Field(default=None, max_length=50)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    firm_name: str
    name: str
    email: EmailStr
    phone: Optional[str]
    created_at: datetime


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class UploadCreate(BaseModel):
    user_id: UUID
    kind: str = Field(default="original", max_length=50)
    parent_upload_id: Optional[UUID] = None
    storage_url: str = Field(min_length=1)
    content_type: Optional[str] = Field(default=None, max_length=100)
    original_filename: Optional[str] = Field(default=None, max_length=255)
    size_bytes: Optional[int] = Field(default=None, ge=0)


class UploadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    kind: str
    parent_upload_id: Optional[UUID]
    storage_url: str
    content_type: Optional[str]
    original_filename: Optional[str]
    size_bytes: Optional[int]
    created_at: datetime
    download_url: Optional[str] = None


class UploadPresignRequest(BaseModel):
    user_id: UUID
    filename: str = Field(min_length=1, max_length=255)
    content_type: Optional[str] = Field(default=None, max_length=100)


class UploadPresignResponse(BaseModel):
    upload_url: str
    method: str
    storage_key: str
    storage_url: str
    required_headers: dict[str, str]
    download_url: Optional[str] = None


class UploadConfirm(BaseModel):
    user_id: UUID
    storage_key: str = Field(min_length=1)
    kind: str = Field(default="original", max_length=50)
    parent_upload_id: Optional[UUID] = None
    content_type: Optional[str] = Field(default=None, max_length=100)
    original_filename: Optional[str] = Field(default=None, max_length=255)
    size_bytes: Optional[int] = Field(default=None, ge=0)


class RoomDimensions(BaseModel):
    width: float
    depth: float
    height: float
    area: float
    unit: str = "m"
    calibration_factor: Optional[float] = None


class DemoWatermarkResponse(BaseModel):
    original: UploadRead
    staged: UploadRead
    dimensions: Optional[RoomDimensions] = None
    ceiling_overlay_base64: Optional[str] = None
    ceiling_corners: Optional[list[list[int]]] = None


class PaymentCreate(BaseModel):
    user_id: UUID
    order_id: Optional[UUID] = None
    status: str = Field(min_length=1, max_length=50)
    stripe_customer_id: Optional[str] = Field(default=None, max_length=255)
    stripe_session_id: Optional[str] = Field(default=None, max_length=255)
    stripe_payment_intent_id: Optional[str] = Field(default=None, max_length=255)
    amount_cents: Optional[int] = Field(default=None, ge=0)
    currency: Optional[str] = Field(default=None, max_length=10)


class PaymentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    order_id: Optional[UUID]
    status: str
    stripe_customer_id: Optional[str]
    stripe_session_id: Optional[str]
    stripe_payment_intent_id: Optional[str]
    amount_cents: Optional[int]
    currency: Optional[str]
    created_at: datetime


class OrderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    status: str
    note: Optional[str]
    image_count: int
    amount_cents: int
    currency: str
    stripe_session_id: Optional[str]
    created_at: datetime
    uploads: list[UploadRead] = Field(default_factory=list)


class OrderListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    note: Optional[str]
    image_count: int
    amount_cents: int
    currency: str
    created_at: datetime


class OrderCheckoutResponse(BaseModel):
    order_id: UUID
    checkout_url: str
    amount_cents: int
    currency: str
    image_count: int


class OrderCheckoutLink(BaseModel):
    checkout_url: str
