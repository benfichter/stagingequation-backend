from uuid import UUID

import logging
import os

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models, schemas
from app.db import get_db
from app.gemini import build_staging_prompt, generate_staged_image
from app.moge import infer_room_dimensions, is_moge_enabled, request_moge_warmup
from app.stripe_utils import get_checkout_urls, get_price_settings, get_stripe
from app.storage import (
    build_object_key,
    build_staged_key,
    create_presigned_get_url,
    create_presigned_put_url,
    get_storage_url,
    put_object_bytes,
)
from app.watermark import apply_watermark

app = FastAPI(title="Staging Equation API")
logger = logging.getLogger(__name__)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}


@app.post("/users", response_model=schemas.UserRead, status_code=status.HTTP_201_CREATED)
def create_user(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.scalar(select(models.User).where(models.User.email == payload.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="email already exists")

    user = models.User(**payload.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/users/{user_id}", response_model=schemas.UserRead)
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    user = db.get(models.User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    return user


@app.get("/users/{user_id}/orders", response_model=list[schemas.OrderListItem])
def list_orders_for_user(user_id: UUID, db: Session = Depends(get_db)):
    user = db.get(models.User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    orders = db.scalars(
        select(models.Order).where(models.Order.user_id == user_id).order_by(models.Order.created_at.desc())
    ).all()
    return [schemas.OrderListItem.model_validate(order) for order in orders]


@app.post("/uploads", response_model=schemas.UploadRead, status_code=status.HTTP_201_CREATED)
def create_upload(payload: schemas.UploadCreate, db: Session = Depends(get_db)):
    user = db.get(models.User, payload.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    upload = models.Upload(**payload.model_dump())
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


@app.post("/uploads/presign", response_model=schemas.UploadPresignResponse)
def presign_upload(payload: schemas.UploadPresignRequest, db: Session = Depends(get_db)):
    user = db.get(models.User, payload.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    content_type = payload.content_type or "application/octet-stream"
    storage_key = build_object_key("uploads", str(payload.user_id), payload.filename)
    upload_url = create_presigned_put_url(storage_key, content_type)
    storage_url = get_storage_url(storage_key)

    download_url = None
    if storage_url.startswith("r2://"):
        download_url = create_presigned_get_url(storage_key)

    return schemas.UploadPresignResponse(
        upload_url=upload_url,
        method="PUT",
        storage_key=storage_key,
        storage_url=storage_url,
        required_headers={"Content-Type": content_type},
        download_url=download_url,
    )


@app.post("/uploads/confirm", response_model=schemas.UploadRead, status_code=status.HTTP_201_CREATED)
def confirm_upload(payload: schemas.UploadConfirm, db: Session = Depends(get_db)):
    user = db.get(models.User, payload.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    storage_url = get_storage_url(payload.storage_key)
    upload = models.Upload(
        user_id=payload.user_id,
        storage_url=storage_url,
        content_type=payload.content_type,
        original_filename=payload.original_filename,
        size_bytes=payload.size_bytes,
        kind=payload.kind,
        parent_upload_id=payload.parent_upload_id,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    download_url = None
    if storage_url.startswith("r2://"):
        download_url = create_presigned_get_url(payload.storage_key)

    return schemas.UploadRead.model_validate(upload).model_copy(update={"download_url": download_url})


@app.post("/demo/watermark", response_model=schemas.DemoWatermarkResponse)
async def demo_watermark(
    user_id: UUID = Form(...),
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
    room_type: str | None = Form(None),
    style: str | None = Form(None),
    calibration_height_m: float | None = Form(None),
    db: Session = Depends(get_db),
):
    user = db.get(models.User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    content_type = file.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty file")

    dimensions = None
    ceiling_overlay_base64 = None
    ceiling_corners = None
    if is_moge_enabled():
        try:
            moge_payload = await run_in_threadpool(
                infer_room_dimensions, contents, calibration_height_m
            )
            if moge_payload:
                dimensions = (
                    moge_payload.get("dimensions")
                    if isinstance(moge_payload, dict) and "dimensions" in moge_payload
                    else moge_payload
                )
                if isinstance(moge_payload, dict):
                    ceiling_overlay_base64 = moge_payload.get("ceiling_overlay_base64")
                    ceiling_corners = moge_payload.get("ceiling_corners")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    original_key = build_object_key("uploads", str(user_id), file.filename)
    put_object_bytes(original_key, contents, content_type)
    original_upload = models.Upload(
        user_id=user_id,
        storage_url=get_storage_url(original_key),
        content_type=content_type,
        original_filename=file.filename,
        size_bytes=len(contents),
        kind="original",
    )
    db.add(original_upload)
    db.flush()

    prompt_text = build_staging_prompt(prompt, room_type, style)
    try:
        generated_bytes = await generate_staged_image(contents, content_type, prompt_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if not generated_bytes:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Gemini returned no image data")

    try:
        staged_bytes = apply_watermark(generated_bytes)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Watermark failed") from exc
    staged_key = build_staged_key(str(user_id))
    staged_content_type = "image/jpeg"
    put_object_bytes(staged_key, staged_bytes, staged_content_type)
    staged_upload = models.Upload(
        user_id=user_id,
        storage_url=get_storage_url(staged_key),
        content_type=staged_content_type,
        original_filename=f"staged_{file.filename or 'image.jpg'}",
        size_bytes=len(staged_bytes),
        kind="staged",
        parent_upload_id=original_upload.id,
    )
    db.add(staged_upload)
    db.commit()
    db.refresh(original_upload)
    db.refresh(staged_upload)

    original_download_url = None
    staged_download_url = None
    if original_upload.storage_url.startswith("r2://"):
        original_download_url = create_presigned_get_url(original_key)
    if staged_upload.storage_url.startswith("r2://"):
        staged_download_url = create_presigned_get_url(staged_key)

    original_read = schemas.UploadRead.model_validate(original_upload).model_copy(
        update={"download_url": original_download_url}
    )
    staged_read = schemas.UploadRead.model_validate(staged_upload).model_copy(
        update={"download_url": staged_download_url}
    )

    return schemas.DemoWatermarkResponse(
        original=original_read,
        staged=staged_read,
        dimensions=schemas.RoomDimensions.model_validate(dimensions) if dimensions else None,
        ceiling_overlay_base64=ceiling_overlay_base64,
        ceiling_corners=ceiling_corners,
    )


@app.post("/payments", response_model=schemas.PaymentRead, status_code=status.HTTP_201_CREATED)
def create_payment(payload: schemas.PaymentCreate, db: Session = Depends(get_db)):
    user = db.get(models.User, payload.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    payment = models.Payment(**payload.model_dump())
    db.add(payment)
    db.commit()
    db.refresh(payment)
    return payment


@app.post("/moge/warm")
def warm_moge():
    if not is_moge_enabled():
        return {"status": "disabled"}

    try:
        return request_moge_warmup()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@app.post("/orders/checkout", response_model=schemas.OrderCheckoutResponse)
async def create_order_checkout(
    user_id: UUID = Form(...),
    files: list[UploadFile] = File(...),
    note: str | None = Form(None),
    db: Session = Depends(get_db),
):
    user = db.get(models.User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="at least one image is required")

    uploads: list[models.Upload] = []
    for file in files:
        content_type = file.content_type or "application/octet-stream"
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="all files must be images")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty file")

        storage_key = build_object_key("orders", str(user_id), file.filename)
        put_object_bytes(storage_key, contents, content_type)
        upload = models.Upload(
            user_id=user_id,
            storage_url=get_storage_url(storage_key),
            content_type=content_type,
            original_filename=file.filename,
            size_bytes=len(contents),
            kind="order_original",
        )
        db.add(upload)
        uploads.append(upload)

    price_per_image, currency = get_price_settings()
    image_count = len(uploads)
    amount_cents = price_per_image * image_count

    clean_note = note.strip() if note else None
    order = models.Order(
        user_id=user_id,
        status="pending_payment",
        note=clean_note,
        image_count=image_count,
        amount_cents=amount_cents,
        currency=currency,
    )
    db.add(order)
    db.flush()

    for upload in uploads:
        db.add(models.OrderItem(order_id=order.id, upload_id=upload.id))

    stripe_client = get_stripe()
    success_url, cancel_url = get_checkout_urls()
    try:
        session = stripe_client.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": currency,
                        "product_data": {
                            "name": "Virtual staging image",
                            "description": "Per-image staging service",
                        },
                        "unit_amount": price_per_image,
                    },
                    "quantity": image_count,
                }
            ],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "order_id": str(order.id),
                "user_id": str(user_id),
                "image_count": str(image_count),
            },
        )
    except Exception as exc:  # stripe.error.StripeError is not imported here
        logger.exception("Stripe checkout failed")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"stripe checkout failed: {exc}"
        ) from exc

    order.stripe_session_id = session.id
    payment = models.Payment(
        user_id=user_id,
        order_id=order.id,
        status="pending",
        stripe_session_id=session.id,
        amount_cents=amount_cents,
        currency=currency,
    )
    db.add(payment)
    db.commit()
    db.refresh(order)

    return schemas.OrderCheckoutResponse(
        order_id=order.id,
        checkout_url=session.url,
        amount_cents=amount_cents,
        currency=currency,
        image_count=image_count,
    )


@app.get("/orders/{order_id}", response_model=schemas.OrderRead)
def get_order(order_id: UUID, db: Session = Depends(get_db)):
    order = db.get(models.Order, order_id)
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="order not found")
    items = db.scalars(
        select(models.OrderItem)
        .where(models.OrderItem.order_id == order_id)
        .order_by(models.OrderItem.created_at.asc())
    ).all()
    uploads: list[schemas.UploadRead] = []
    for item in items:
        upload = item.upload
        if not upload:
            continue
        download_url = None
        if upload.storage_url.startswith("r2://"):
            key_parts = upload.storage_url[len("r2://") :].split("/", 1)
            if len(key_parts) == 2:
                download_url = create_presigned_get_url(key_parts[1])
        uploads.append(
            schemas.UploadRead.model_validate(upload).model_copy(update={"download_url": download_url})
        )
    return schemas.OrderRead.model_validate(order).model_copy(update={"uploads": uploads})


@app.post("/orders/{order_id}/checkout", response_model=schemas.OrderCheckoutLink)
def resume_order_checkout(order_id: UUID, db: Session = Depends(get_db)):
    order = db.get(models.Order, order_id)
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="order not found")

    if order.status != "pending_payment":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="order is not pending payment")

    if not order.stripe_session_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no checkout session available")

    stripe_client = get_stripe()
    try:
        session = stripe_client.checkout.Session.retrieve(order.stripe_session_id)
    except Exception as exc:  # stripe.error.StripeError is not imported here
        logger.exception("Stripe checkout retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"stripe checkout retrieval failed: {exc}"
        ) from exc

    if not session.url:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="checkout session is not available")

    return schemas.OrderCheckoutLink(checkout_url=session.url)


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    if not sig_header:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="missing stripe signature")

    stripe_client = get_stripe()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="webhook secret not set")

    try:
        event = stripe_client.Webhook.construct_event(payload, sig_header, webhook_secret)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid webhook signature") from exc

    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        session_id = data_object.get("id")
        metadata = data_object.get("metadata") or {}
        order_id = metadata.get("order_id")
        if order_id:
            try:
                order_uuid = UUID(order_id)
            except ValueError:
                order_uuid = None
            if order_uuid:
                order = db.get(models.Order, order_uuid)
                if order:
                    order.status = "paid"
                    order.stripe_session_id = session_id
                    order.stripe_payment_intent_id = data_object.get("payment_intent")

        payment = db.scalar(select(models.Payment).where(models.Payment.stripe_session_id == session_id))
        if payment:
            payment.status = "paid"
            payment.stripe_customer_id = data_object.get("customer")
            payment.stripe_payment_intent_id = data_object.get("payment_intent")
        db.commit()

    elif event_type == "checkout.session.expired":
        session_id = data_object.get("id")
        payment = db.scalar(select(models.Payment).where(models.Payment.stripe_session_id == session_id))
        if payment:
            payment.status = "expired"
        order = db.scalar(select(models.Order).where(models.Order.stripe_session_id == session_id))
        if order:
            order.status = "expired"
        db.commit()

    return {"received": True}
