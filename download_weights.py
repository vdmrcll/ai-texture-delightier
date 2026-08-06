"""Download and verify the ONNX models used by Texture Delighter.

The model files are distributed separately from the application because they
are large. Set DELIGHTER_MODEL_BASE_URL to use a mirror; the default points
to the latest GitHub release assets.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = APP_DIR / "weights"
DEFAULT_BASE_URL = (
    "https://github.com/vdmrcll/ai-texture-delightier/releases/latest/download"
)

# SHA-256 values for the model files currently shipped with v0.2-beta.
MODELS = {
    "delighter_model_fp16.onnx": {
        "sha256": "a20dba3aa81e90594029c2870da7a5a16ece52b2c75e5074146e71a4b3086a59",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    actual = sha256(path)
    if actual != expected:
        print(f"Invalid checksum for {path.name}; expected {expected}, got {actual}.")
        return False
    return True


def download(name: str, destination: Path, base_url: str) -> None:
    url = f"{base_url.rstrip('/')}/{name}"
    print(f"Downloading {name}...")
    request = Request(url, headers={"User-Agent": "Texture-Delighter/0.3"})
    with urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{name}.", suffix=".part", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            received = 0
            try:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
                    received += len(chunk)
                    if total:
                        print(f"\r  {received / total:.0%}", end="", flush=True)
                if total:
                    print()
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
    temporary_path.replace(destination)


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("DELIGHTER_MODEL_BASE_URL", DEFAULT_BASE_URL)

    try:
        for name, metadata in MODELS.items():
            destination = WEIGHTS_DIR / name
            if verify(destination, metadata["sha256"]):
                print(f"Verified {name}.")
                continue

            # Replace incomplete or corrupt files only after a complete
            # download has been written to a temporary file.
            download(name, destination, base_url)
            if not verify(destination, metadata["sha256"]):
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded model failed verification: {name}")
            print(f"Verified {name}.")
    except Exception as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        return 1

    print("Model weights are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
