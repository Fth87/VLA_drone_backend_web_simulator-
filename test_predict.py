import io
import json
import urllib.request
from urllib.error import HTTPError

from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8000"
INSTRUCTION = "Selesaikan lintasan"


def make_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), (30, 80, 120))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 112), (224, 112)], fill=(180, 200, 220), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def multipart_body(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, (filename, data, content_type) in files.items():
        body += (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main():
    # Health check
    with urllib.request.urlopen(f"{BASE_URL}/health") as r:
        print("Health:", json.loads(r.read()))

    # Predict
    body, content_type = multipart_body(
        fields={"language_instruction": INSTRUCTION},
        files={"image": ("frame.jpg", make_image_bytes(), "image/jpeg")},
    )
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
    except HTTPError as e:
        print(f"Error {e.code}:", e.read().decode())
        return

    print("Action:", result["action"])
    print("Timestamp:", result["timestamp"])


if __name__ == "__main__":
    main()
