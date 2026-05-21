import os
import sys
from pathlib import Path

import uvicorn


# Set working directory to this file's directory and ensure it's on sys.path
BASE_DIR = Path(__file__).resolve().parent
os.chdir(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR))


def main() -> None:
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(BASE_DIR)],
    )


if __name__ == "__main__":
    main()
