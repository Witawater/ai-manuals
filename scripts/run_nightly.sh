#!/bin/bash
while true; do
  echo "$(date) – running nightly QA"
  python tests/drain_eval.py >> /mnt/data/manual_eval.log 2>&1
  sleep 86400
done
