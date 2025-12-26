from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def apply_watermark(image_bytes: bytes, text: str = "DEMO") -> bytes:
    with Image.open(BytesIO(image_bytes)) as image:
        base = image.convert("RGBA")

    width, height = base.size
    font_size = max(32, min(width, height) // 5)
    font = _load_font(font_size)

    diag = int((width**2 + height**2) ** 0.5)
    watermark = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)

    text_width, text_height = _text_size(draw, text, font)
    step_x = text_width + font_size
    step_y = text_height + font_size

    for y in range(0, diag, step_y):
        for x in range(0, diag, step_x):
            shadow_pos = (x + 3, y + 3)
            draw.text(shadow_pos, text, font=font, fill=(0, 0, 0, 180))
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))

    rotated = watermark.rotate(-30, expand=True)
    rx, ry = rotated.size
    left = max(0, (rx - width) // 2)
    top = max(0, (ry - height) // 2)
    cropped = rotated.crop((left, top, left + width, top + height))

    combined = Image.alpha_composite(base, cropped).convert("RGB")
    output = BytesIO()
    combined.save(output, format="JPEG", quality=90)
    return output.getvalue()
