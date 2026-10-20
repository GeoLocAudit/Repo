"""E2: does the network location change the cited sources? Compare sessions (net_*) per engine.

For each question, cited-domain Jaccard between the two twins of a session (noise) and between the
same repeat in two different sessions. Localisation index = twin - between (0 = no network effect).
Bootstrap 95% CI over questions.

  python scripts/analyse_network.py net_ca1 net_us_east [net_us_west ...]
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_pilot import jaccard, load  # noqa: E402
from geoaudit.common import RESULTS  # noqa: E402

RNG = np.random.default_rng(42)


def pairs(d):
    out = []
    for qid, g in d[d.cites].groupby('qid'):
        recs = g.to_dict('records')
        for x, y in itertools.combinations(recs, 2):
            if x['session'] == y['session']:
                kind = 'twin'
            elif x['rep'] == y['rep']:
                kind = 'between'
            else:
                continue
            out.append(dict(qid=qid, group=x['group'], kind=kind,
                            sessions='-'.join(sorted([x['session'], y['session']])) if kind == 'between' else x['session'],
                            j=jaccard(x['domains'], y['domains'])))
    return pd.DataFrame(out)


def boot_index(p, n=2000):
    per_q = p.groupby(['qid', 'kind']).j.mean().unstack()
    per_q = per_q.dropna()
    diff = (per_q['twin'] - per_q['between']).to_numpy()
    if len(diff) == 0:
        return np.nan, np.nan, np.nan, 0
    b = [RNG.choice(diff, len(diff)).mean() for _ in range(n)]
    return diff.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(diff)


def main():
    runs = sys.argv[1:] or ['net_ca1', 'net_us_east']
    rows = []
    for engine in ('gemini', 'openai'):
        d = pd.concat([load(r, engine).assign(session=r) for r in runs], ignore_index=True)
        p = pairs(d)
        print(f'\n=== {engine}: sessions {runs}, {len(d)} calls')
        print('mean Jaccard of cited domains:\n' +
              p.pivot_table(index='group', columns='kind', values='j', aggfunc='mean').round(2).to_string())
        for group, g in p.groupby('group'):
            m, lo, hi, n = boot_index(g)
            rows.append(dict(engine=engine, group=group, index=m, lo=lo, hi=hi, questions=n))
        b = p[p.kind == 'between'].groupby(['sessions', 'group']).j.mean().unstack().round(2)
        print('between-session overlap by session pair:\n' + b.to_string())
    r = pd.DataFrame(rows)
    print('\nlocalisation index (twin - between), 95% CI over questions:\n' + r.round(3).to_string(index=False))
    r.to_csv(RESULTS / ('e2_index_' + '_'.join(runs) + '.csv'), index=False)


if __name__ == '__main__':
    main()
