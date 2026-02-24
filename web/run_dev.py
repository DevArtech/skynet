from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "web" / "frontend"


def main() -> int:
    backend_cmd = [sys.executable, "-m", "uvicorn", "web.backend.main:app", "--reload", "--port", "8000"]
    frontend_cmd = ["npm", "run", "dev"]

    backend = subprocess.Popen(backend_cmd, cwd=ROOT)
    frontend = subprocess.Popen(frontend_cmd, cwd=FRONTEND_DIR)
    try:
        backend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.terminate()
        backend.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
