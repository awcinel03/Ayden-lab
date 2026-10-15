"""No tracked file may contain a key file or a key value. Reports locations, never values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_FILES = re.compile(r"(^|/)(\.env(\.local)?|secrets\.toml)$")
KEY_VALUES = re.compile(
    r"sb_secret_[A-Za-z0-9_-]{6,}"
    r"|sb_publishable_[A-Za-z0-9_-]{6,}"
    r"|sbp_[A-Za-z0-9]{20,}"
    r"|eyJ[A-Za-z0-9_-]{20,}\.eyJ"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_no_key_files_are_tracked():
    assert not [p for p in tracked_files() if KEY_FILES.search(p)]


def test_no_key_values_in_tracked_files():
    offenders = []
    for rel in tracked_files():
        path = ROOT / rel
        if path.is_file() and KEY_VALUES.search(path.read_text(errors="replace")):
            offenders.append(rel)
    assert not offenders, f"possible secrets in: {offenders}"
