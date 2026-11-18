#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec "$HOME/miniconda3/envs/cs180/bin/python" app.py
