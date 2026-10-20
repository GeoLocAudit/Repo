"""E1 (declared location, ChatGPT): how much, through which channel, and what changes.

For each arm (e1_natural, e1_search):
 1. Localisation index by question group = twin overlap - between-location overlap of cited domains
    (per question, bootstrap 95% CI over questions), and by pair type: two US cities, US city vs.
    another country, two other countries.
 2. Channel: pairs in which neither answer's queries name its location (backend localisation) vs.
    pairs in which both do (query rewriting); twins restricted the same way.
 3. Place-specific sources: share of answers citing a URL that names the answer's own location
    (city, US state, or the country's domain ending), against a placebo: one naming another of the
    locations. Also: does "none" behave like a US city?

  python scripts/analyse_e1.py
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_pilot import jaccard, load  # noqa: E402
from geoaudit.common import DATA, RESULTS  # noqa: E402

RNG = np.random.default_rng(42)
CCTLD = {'GB': '.uk', 'CA': '.ca', 'AU': '.au', 'IE': '.ie'}
LOC = pd.read_csv(DATA / 'locations.csv').set_index('loc_id')


def forms(name):
    n = name.lower()
    return {n.replace(' ', s) for s in ('-', '', '_', '+', '%20')}


def place_terms(lid):
    r = LOC.loc[lid]
    t = forms(r.city)
    if r.country == 'US':
        t |= forms(r.region)
    else:
        t |= {CCTLD[r.country] + '/'}                 # domain ending, matched on host + '/'
    return t


TERMS = {lid: place_terms(lid) for lid in LOC.index}


def names_place(urls, lid):
    for u in urls:
        m = re.match(r'https?://([^/]+)(.*)', u.lower())
        if not m:
            continue
        s = m.group(1) + '/' + m.group(2)
        if any(t in s for t in TERMS[lid]):
            return True
    return False


def kind_of(a, b):
    ca, cb = (LOC.at[x, 'country'] if x in LOC.index else 'none' for x in (a, b))
    if 'none' in (ca, cb):
        return 'with none'
    if ca == cb == 'US':
        return 'two US cities'
    if 'US' in (ca, cb):
        return 'US city vs other country'
    return 'two other countries'


def pairs(d):
    out = []
    for qid, g in d[d.cites].groupby('qid'):
        recs = g.to_dict('records')
        for x, y in itertools.combinations(recs, 2):
            same = x['loc_id'] == y['loc_id']
            if not same and x['rep'] != y['rep']:
                continue
            out.append(dict(qid=qid, group=x['group'], twin=same, kind='twin' if same else kind_of(x['loc_id'], y['loc_id']),
                            place_q=('both' if x['loc_in_query'] and y['loc_in_query'] else
                                     'neither' if not x['loc_in_query'] and not y['loc_in_query'] else 'one'),
                            real=x['loc_id'] != 'none' and y['loc_id'] != 'none',
                            j=jaccard(x['domains'], y['domains'])))
    return pd.DataFrame(out)


def index(p, between_mask, twin_mask=None, n=2000):
    """Per question: mean twin overlap - mean between overlap; bootstrap CI over questions."""
    tw = p[p.twin & (twin_mask if twin_mask is not None else True)].groupby('qid').j.mean()
    bt = p[between_mask].groupby('qid').j.mean()
    diff = (tw - bt).dropna().to_numpy()
    if len(diff) < 5:
        return dict(index=np.nan, lo=np.nan, hi=np.nan, questions=len(diff))
    b = [RNG.choice(diff, len(diff)).mean() for _ in range(n)]
    return dict(index=diff.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), questions=len(diff))


def main():
    out1, out2, out3 = [], [], []
    for arm in ('e1_natural', 'e1_search'):
        d = load(arm)
        d['own_place'] = [names_place(u, l) if l in LOC.index else np.nan for u, l in zip(d.urls, d.loc_id)]
        others = {l: [o for o in LOC.index if o != l and LOC.at[o, 'country'] == LOC.at[l, 'country']]
                  or [o for o in LOC.index if o != l] for l in LOC.index}
        d['other_place'] = [np.mean([names_place(u, o) for o in others[l]]) if l in LOC.index else np.nan
                            for u, l in zip(d.urls, d.loc_id)]
        p = pairs(d)
        p = p[p.real | p.twin]
        for group, g in p.groupby('group'):
            real_between = ~g.twin & g.real
            out1.append(dict(arm=arm, group=group, pairs='all locations', **index(g, real_between)))
            for k in ('two US cities', 'US city vs other country', 'two other countries'):
                out1.append(dict(arm=arm, group=group, pairs=k, **index(g, real_between & (g.kind == k))))
            for pq in ('neither', 'both'):
                out2.append(dict(arm=arm, group=group, place_in_queries=pq,
                                 **index(g, real_between & (g.place_q == pq), g.place_q == pq),
                                 between_pairs=int((real_between & (g.place_q == pq)).sum())))
        real = d[d.loc_id.isin(LOC.index) & d.cites]
        for (group, us), g in real.groupby(['group', real.loc_id.map(LOC.country).eq('US')]):
            out3.append(dict(arm=arm, group=group, locations='US cities' if us else 'other countries',
                             own_place=g.own_place.mean() * 100, other_place_placebo=g.other_place.mean() * 100,
                             answers=len(g)))
    r1, r2, r3 = (pd.DataFrame(x) for x in (out1, out2, out3))
    pd.set_option('display.width', 160)
    print('1. Localisation index (twin - between, cited domains), 95% CI over questions\n' +
          r1.round(3).to_string(index=False))
    print('\n2. Channel: pairs whose queries name neither / both locations\n' + r2.round(3).to_string(index=False))
    print('\n3. Answers citing a source that names the own location vs another location (placebo), %\n' +
          r3.round(1).to_string(index=False))
    for name, r in (('e1_index', r1), ('e1_channel', r2), ('e1_place_sources', r3)):
        r.to_csv(RESULTS / f'{name}.csv', index=False)


if __name__ == '__main__':
    main()
