#!/usr/bin/env bash
# Everything that runs from the native connection (no VPN): E2 Canada session, then E1.
# E1 must finish before any VPN switch, so that ChatGPT's network location is constant within E1.
#   bash scripts/run_native.sh > results/run_native.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../AeoCiteAnalysis/.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
E1_LOCS="$($PY -c "import pandas as pd; print(','.join(list(pd.read_csv('data/locations.csv').loc_id) + ['none']))")"
quiet() { grep --line-buffered -v -i "pydantic\|serializ\|return self\|warnings.warn"; }

echo "=== $(date '+%F %T') E2 session net_ca1 (native, Canada)"
$PY scripts/run.py --engine gemini --run net_ca1 --locs none --reps 2 --allow-gemini --workers 6 2>&1 | quiet &
$PY scripts/run.py --engine openai --run net_ca1 --locs none --reps 2 --model gpt-5.4-mini \
    --template "Search the web: {q}" --workers 6 2>&1 | quiet
wait

echo "=== $(date '+%F %T') E1 declared location: natural and search arms (gpt-5.4-mini)"
$PY scripts/run.py --engine openai --run e1_natural --locs "$E1_LOCS" --reps 2 --model gpt-5.4-mini --workers 10 2>&1 | quiet &
$PY scripts/run.py --engine openai --run e1_search --locs "$E1_LOCS" --reps 2 --model gpt-5.4-mini \
    --template "Search the web: {q}" --workers 10 2>&1 | quiet
wait

echo "=== $(date '+%F %T') retry pass (skips calls already saved)"
$PY scripts/run.py --engine gemini --run net_ca1 --locs none --reps 2 --allow-gemini 2>&1 | quiet
$PY scripts/run.py --engine openai --run net_ca1 --locs none --reps 2 --model gpt-5.4-mini --template "Search the web: {q}" 2>&1 | quiet
$PY scripts/run.py --engine openai --run e1_natural --locs "$E1_LOCS" --reps 2 --model gpt-5.4-mini 2>&1 | quiet
$PY scripts/run.py --engine openai --run e1_search --locs "$E1_LOCS" --reps 2 --model gpt-5.4-mini --template "Search the web: {q}" 2>&1 | quiet
echo "=== $(date '+%F %T') native runs finished"
