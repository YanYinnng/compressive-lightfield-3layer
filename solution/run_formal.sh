#!/usr/bin/env bash
# Formal test on the H100: unzip the released test archive, solve both branches, format check.
# The TA now collects the folder path, so no zip is written (pass --zip to get one).
# usage: bash run_formal.sh <test_zip_or_unpacked_dir> <work_dir> [group_name] [extra submit_v3.py args]
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ZIP="$1"; WORK="$2"; GROUP="${3:-group_example}"
shift $(( $# < 3 ? $# : 3 ))
date '+[start] %F %T'
mkdir -p "$WORK/input"
if [[ -d "$ZIP" ]]; then INPUT="$ZIP"; else
  python -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$ZIP" "$WORK/input"; INPUT="$WORK/input"; fi
find "$INPUT" -name test.json | grep -q . || { echo 'test.json not found' >&2; exit 2; }
# All scenes (test_input/<scene>/test.json) go into one submission and one index.json.
python -u -B "$HERE/submit_v3.py" --index "$INPUT" --out "$WORK/$GROUP" "$@"
# The report (incl. the off-axis analysis) is written before the test and copied verbatim.
if [[ -f "$HERE/report_final.md" ]]; then cp "$HERE/report_final.md" "$WORK/$GROUP/report.md"; fi
chmod -R a+rX "$WORK"
FOLDER="$(cd -- "$WORK/$GROUP" && pwd)"
# Written last and atomically: the CPU watcher submits as soon as this file appears.
echo "$FOLDER" > "$WORK/$GROUP.txt.part" && mv "$WORK/$GROUP.txt.part" "$WORK/$GROUP.txt"
date '+[ready] %F %T'
# submit_v3 only ran its fast check; the full TA checker (decodes every PNG, ~2.5 min per 1000
# samples) runs now, off the critical path. Exit 5 if it rejects the already published folder.
FULL=0; PREV=""; for A in "$@"; do [[ "$PREV" = --ta-check && "$A" = full ]] && FULL=1; PREV="$A"; done
if (( ! FULL )) && [[ -f "$WORK/${GROUP}_all_test.json" ]]; then
  date '+[post-check] TA format check started %F %T'
  CHECKER="$HERE/../testing_v3/check_submission.py"; [[ -f "$CHECKER" ]] || CHECKER="$HERE/check_submission.py"  # release / packaged code/
  if python -B "$CHECKER" --index "$WORK/${GROUP}_all_test.json" --submission "$WORK/$GROUP/index.json"; then
    date '+[post-check] PASS %F %T'
  else
    date '+[post-check] FAIL: the TA checker rejected the published folder %F %T'; exit 5
  fi
fi
