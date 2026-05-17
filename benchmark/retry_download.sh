#!/bin/bash
# 自动重试下载，直到成功
# Usage: bash retry_download.sh <repo_id>

set -e

REPO_ID="$1"
MAX_RETRIES=20
SLEEP_SEC=10

if [ -z "$REPO_ID" ]; then
    echo "Usage: bash retry_download.sh <repo_id>"
    exit 1
fi

echo "Downloading: $REPO_ID (max $MAX_RETRIES retries)"
echo "Mirror: ${HF_ENDPOINT:-https://hf-mirror.com}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

for i in $(seq 1 $MAX_RETRIES); do
    echo "--- Attempt $i/$MAX_RETRIES at $(date '+%H:%M:%S') ---"
    if hf download "$REPO_ID" 2>&1; then
        echo "✅ Download complete: $REPO_ID"
        exit 0
    fi
    echo "❌ Attempt $i failed, retrying in ${SLEEP_SEC}s..."
    sleep $SLEEP_SEC
done

echo "❌ All $MAX_RETRIES attempts failed for: $REPO_ID"
exit 1
