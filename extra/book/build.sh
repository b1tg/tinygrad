#!/bin/bash
# Thin wrapper around build.py
set -e
BOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$BOOK_DIR/build.py" "$@"
