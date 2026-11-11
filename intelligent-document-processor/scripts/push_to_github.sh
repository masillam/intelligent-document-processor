#!/usr/bin/env bash
# Usage: ./scripts/push_to_github.sh [repo-name]   (requires: git + GitHub CLI `gh auth login`)
set -euo pipefail
REPO="${1:-intelligent-document-processor}"
[ -d .git ] || git init -b main
git add .
git commit -m "Initial commit: Intelligent Document Processor" || true
gh repo create "$REPO" --public --source=. --remote=origin --push
