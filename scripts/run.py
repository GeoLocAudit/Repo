"""Send questions x locations x repeats to an engine, one JSON file per call.

  data/raw/<engine>/<run>/<qid>__<loc_id>__r<rep>.json   (existing files are skipped, so runs resume)

Locations come from data/locations.csv plus two controls:
  none    no location sent (the API's own default)
  blank   ChatGPT only: user_location {"type": "approximate"} with no fields
Calls are shuffled, so twins and locations of a question are spread over the same time window.

  python scripts/run.py --engine openai --run pilot1 --pilot --reps 2
  python scripts/run.py --engine openai --run pilot1f --pilot --reps 2 --followup
  python scripts/run.py --engine openai --run main1 --reps 2
  python scripts/run.py --engine gemini --run net_us_east --locs none --allow-gemini   (E2, behind the VPN)
Gemini shares the ~1,500/day grounding quota with the daily replay of the other project; it is
refused unless --allow-gemini is given (run it only on days without the replay).
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoaudit.common import DATA, RAW, SEED  # noqa: E402
from geoaudit.engines import ASK, FOLLOWUP  # noqa: E402


def design(engine, pilot, reps, n_questions, only=None, sample=0):
    q = pd.read_csv(DATA / 'questions.csv')
    loc = pd.read_csv(DATA / 'locations.csv')
    if pilot:
        q, loc = q[q.pilot == 1], loc[loc.pilot == 1]
    if n_questions:
        q = q.groupby('group', group_keys=False).apply(lambda g: g.head(n_questions // 3))
    if sample:                                  # random, stratified by group
        q = pd.concat([g.sample(sample // 3, random_state=SEED) for _, g in q.groupby('group')])
    locs = [(r.loc_id, r._asdict()) for r in loc.itertuples(index=False)] + [('none', None)]
    if engine == 'openai':
        locs.append(('blank', {}))
    if only:
        locs = [x for x in locs if x[0] in only]
    calls = [(r.qid, r.question, lid, l, rep) for r in q.itertuples() for lid, l in locs for rep in range(1, reps + 1)]
    random.Random(SEED).shuffle(calls)
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=list(ASK), required=True)
    ap.add_argument('--run', required=True, help='run name, e.g. pilot1')
    ap.add_argument('--pilot', action='store_true', help='pilot questions and locations only')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--n', type=int, default=0, help='limit to n questions (balanced over groups)')
    ap.add_argument('--sample', type=int, default=0, help='random sample of n questions, stratified by group')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--model', default=None, help='ChatGPT model (default gpt-5.4-nano)')
    ap.add_argument('--template', default='{q}', help='prompt template, e.g. "Search the web: {q}"')
    ap.add_argument('--followup', action='store_true', help='second turn if the answer cites nothing')
    ap.add_argument('--locs', default='', help='comma list of loc_ids to keep, e.g. none (network-location sessions)')
    ap.add_argument('--allow-gemini', action='store_true')
    a = ap.parse_args()
    if a.engine == 'gemini' and not a.allow_gemini:
        sys.exit('Gemini refused: shares the grounding quota with the daily replay. Use --allow-gemini on a free day.')

    out = RAW / a.engine / a.run
    out.mkdir(parents=True, exist_ok=True)
    calls = [c for c in design(a.engine, a.pilot, a.reps, a.n, set(filter(None, a.locs.split(','))), a.sample)
             if not (out / f'{c[0]}__{c[2]}__r{c[4]}.json').exists()]
    print(f'{a.engine}/{a.run}: {len(calls)} calls to make -> {out}')
    done, fails, lock, t_start = [0], [0], threading.Lock(), time.time()

    def one(c):
        qid, question, lid, l, rep = c
        t0 = time.time()
        try:
            res = ASK[a.engine](a.template.format(q=question), l, FOLLOWUP if a.followup else None,
                               a.model if a.engine == 'openai' else None)
        except Exception as e:
            with lock:
                fails[0] += 1
            print(f'  {qid}/{lid}/r{rep}: {type(e).__name__}: {str(e)[:150]}')
            return
        res.update(qid=qid, question=question, prompt=a.template.format(q=question), loc_id=lid, rep=rep, engine=a.engine, run=a.run,
                   time=datetime.now().isoformat(timespec='seconds'), seconds=round(time.time() - t0, 1))
        (out / f'{qid}__{lid}__r{rep}.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f'  {done[0]}/{len(calls)}  {(time.time() - t_start) / 60:.1f} min')

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, calls))
    print(f'done {done[0]}, failed {fails[0]}, {(time.time() - t_start) / 60:.1f} min')


if __name__ == '__main__':
    main()
