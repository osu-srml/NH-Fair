#!/usr/bin/env bash
set -euo pipefail
python -m release_benchmark.cli.zeroshot "$@"
