#!/usr/bin/env bash
set -euo pipefail

# Install autoresearch runtime dependencies (excluding torch) into a target Python env.
#
# Usage:
#   bash scripts/install_autoresearch_runtime_deps.sh
#
# Optional environment variables:
#   ROOT_DIR=/path/to/autoresearch
#   PYTHON_BIN=python

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[runtime-deps] ERROR: Python interpreter '${PYTHON_BIN}' not found."
  exit 1
fi

mapfile -t RUNTIME_DEPS < <("${PYTHON_BIN}" - "${ROOT_DIR}/pyproject.toml" <<'PY'
import re
import sys
from pathlib import Path

pyproject_path = Path(sys.argv[1])
text = pyproject_path.read_text(encoding="utf-8")
deps = []

tomllib = None
try:
    import tomllib as _tomllib  # Python 3.11+
    tomllib = _tomllib
except ModuleNotFoundError:
    tomllib = None

if tomllib is not None:
    try:
        data = tomllib.loads(text)
        deps = data.get("project", {}).get("dependencies", [])
    except Exception:
        deps = []

if not deps:
    in_deps_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_deps_block:
            if line == "dependencies = [":
                in_deps_block = True
            continue
        if line.startswith("]"):
            break
        if not line or line.startswith("#"):
            continue
        if line[0] in {"'", '"'}:
            deps.append(line.rstrip(",").strip().strip('"').strip("'"))

for dep in deps:
    dep = dep.strip()
    if not dep:
        continue
    pkg_name = re.split(r"[<>=!~\[\] ;]", dep, maxsplit=1)[0].lower()
    if pkg_name == "torch":
        continue
    print(dep)
PY
)

if [[ "${#RUNTIME_DEPS[@]}" -eq 0 ]]; then
  echo "[runtime-deps] ERROR: no non-torch runtime dependencies found in pyproject.toml"
  exit 1
fi

echo "[runtime-deps] Installing ${#RUNTIME_DEPS[@]} runtime dependencies into ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install "${RUNTIME_DEPS[@]}"
