from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models, schemas
from app.db import get_db
from app.gemini import build_staging_prompt, generate_staged_image
from app.moge import infer_room_dimensions, is_moge_enabled
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
