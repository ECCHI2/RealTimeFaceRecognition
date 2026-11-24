import sys
import subprocess
from pathlib import Path


def ensure_yolo_environment():
    project_path = Path(__file__).resolve().parent.parent

    python_path = (
        project_path / ".venv_yolo" / "Scripts" / "python.exe"
    )

    if not python_path.exists():
        raise RuntimeError("YOLO environment not found!")

    if Path(sys.executable).resolve() != python_path.resolve():
        exit_code = subprocess.call([
            str(python_path),
            "-u",
            *sys.argv
        ])

        raise SystemExit(exit_code)