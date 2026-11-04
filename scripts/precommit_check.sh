#!/usr/bin/env bash
# Read-only safety check before you commit: lists what would go to GitHub and flags private or risky files. It changes nothing and never reads .env.
# Usage: scripts/precommit_check.sh        (run from anywhere in the repo)
cd "$(git rev-parse --show-toplevel)" || exit 1
bad=0
files=$( { git diff --name-only HEAD; git diff --name-only --cached; git ls-files --others --exclude-standard; } | sort -u )
echo "Files that would be committed (changed + new, ignored files excluded):"; echo "$files" | sed 's/^/  /'
echo
flag() { echo "  !! $1"; bad=1; }
echo "Checks:"
for f in $files; do
  [ -f "$f" ] || continue
  case "$f" in
    .env|.env.*|*/.env|*/.env.*) [ "$f" = ".env.example" ] || flag "env file: $f" ;;
    assets/*|backend/storage/*|*.mov|*.MOV|*.pem|*.key|*.p12|deploy/tls/*|CLAUDE.md|.claude/*) flag "private or local file: $f" ;;
  esac
  [ "$(wc -c < "$f")" -gt 5000000 ] && flag "bigger than 5 MB: $f ($(du -h "$f" | cut -f1))"
  case "$f" in *.png|*.jpg|*.jpeg|*.glb|*.mp4|*.pdf|*.npz|*.stl|*.ico|*.woff*) continue ;; esac
  if grep -I -n -E "(sk|rk|pk)[-_][A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|MODEL_API_KEY *= *[A-Za-z0-9_-]{16,}" "$f" >/dev/null 2>&1; then flag "looks like a secret in $f (open it and check; value not shown)"; fi
  if grep -I -n -E "/Users/[a-z]|\[date\]|\[contact email\]|\[backend host\]|\[governing law\]" "$f" >/dev/null 2>&1; then flag "personal path or unfilled placeholder in $f"; fi
done
[ "$bad" = 0 ] && echo "  none found" || echo "Fix the lines marked !! before you commit."
exit $bad
