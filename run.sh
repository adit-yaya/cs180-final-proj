#!/bin/sh
set -eu
cd "$(dirname "$0")"
ZOOM_PYTHON=python3
if [ -x .venv/bin/python ]; then
    ZOOM_PYTHON=.venv/bin/python
fi
if command -v ollama >/dev/null 2>&1 && ! "$ZOOM_PYTHON" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)' >/dev/null 2>&1; then
    nohup ollama serve >"${TMPDIR:-/tmp}/cs180-final-proj-ollama.log" 2>&1 &
    echo "Starting local Ollama service for the prompt director."
fi
exec "$ZOOM_PYTHON" app.py
