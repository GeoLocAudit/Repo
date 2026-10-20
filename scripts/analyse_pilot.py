"""Summarise a run: search and clarification rates, location in queries, and citation overlap.

Per call (final turn = the follow-up turn if one was sent):
  searched      the model ran at least one web search (either turn)
  cites         the final answer cites at least one URL
  asks_location the first answer cites nothing and asks for a ZIP code / postcode / city
  loc_in_query  the location's city or region appears in one of its search queries
Overlap: Jaccard of cited domains between two calls of the same question that both cite something,
  twin = same location, other repeat (noise); between = different real locations, same repeat.
  The location effect is between minus twin overlap (lower overlap between locations = more localised).

  python scripts/analyse_pilot.py pilot1 [pilot1f gemini:pilot1 ...]   (engine:run, default openai)
"""
import itertools
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoaudit.common import DATA, RAW, RESULTS, clean_url, domain_of  # noqa: E402

ASK = re.compile(r'\b(zip|zip code|postcode|post code|postal code|your (city|location|area|address|neighbo[u]?rhood))\b', re.I)


def load(run, engine='openai'):
    loc = pd.read_csv(DATA / 'locations.csv').set_index('loc_id')
    q = pd.read_csv(DATA / 'questions.csv').set_index('qid')
    rows = []
    for f in sorted((RAW / engine / run).glob('*.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        fu = j.get('followup')
        final = fu or j
        queries = j['queries'] + (fu['queries'] if fu else [])
        names = []
        if j['loc_id'] in loc.index:
            names = [loc.at[j['loc_id'], 'city'], loc.at[j['loc_id'], 'region']]
        cited = list(dict.fromkeys(clean_url(u) for u in final['cited']))
        rows.append(dict(qid=j['qid'], group=q.at[j['qid'], 'group'], topic=q.at[j['qid'], 'topic'],
                         loc_id=j['loc_id'], rep=j['rep'], searched=bool(queries), cites=bool(cited),
                         asks_location=not j['cited'] and bool(ASK.search(j['answer'])),
                         followup_sent=fu is not None,
                         loc_in_query=any(n.lower() in ' '.join(queries).lower() for n in names if isinstance(n, str)),
                         n_queries=len(queries), domains=sorted({domain_of(u) for u in cited}), urls=cited))
    return pd.DataFrame(rows)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float('nan')


def overlap(d):
    real = d[~d.loc_id.isin(['none', 'blank'])]
    out = []
    for qid, g in real[real.cites].groupby('qid'):
        recs = g.to_dict('records')
        for x, y in itertools.combinations(recs, 2):
            if x['loc_id'] == y['loc_id']:
                kind = 'twin'
            elif x['rep'] == y['rep']:
                kind = 'between'
            else:
                continue
            out.append(dict(qid=qid, group=x['group'], kind=kind, j_domain=jaccard(x['domains'], y['domains']),
                            j_url=jaccard(x['urls'], y['urls'])))
    return pd.DataFrame(out)


def main():
    RESULTS.mkdir(exist_ok=True)
    for arg in sys.argv[1:] or ['pilot1']:
        engine, run = arg.split(':') if ':' in arg else ('openai', arg)
        d = load(run, engine)
        run = f'{engine}_{run}' if engine != 'openai' else run
        print(f'\n=== {run}: {len(d)} calls, {d.qid.nunique()} questions, {d.loc_id.nunique()} location settings')
        rates = d.groupby('group')[['searched', 'cites', 'asks_location', 'followup_sent']].mean().mul(100).round(0)
        rates['loc_in_query'] = (d[d.searched & ~d.loc_id.isin(['none', 'blank'])]
                                 .groupby('group').loc_in_query.mean().mul(100).round(0))
        print('rates by group (%):\n' + rates.to_string())
        print('cites by location (%):\n' + d.groupby('loc_id').cites.mean().mul(100).round(0).to_string())
        o = overlap(d)
        if len(o):
            ov = o.pivot_table(index='group', columns='kind', values=['j_domain', 'j_url'], aggfunc='mean').round(2)
            n = o.pivot_table(index='group', columns='kind', values='j_url', aggfunc='size')
            print('citation overlap (Jaccard), twin = same location, between = different locations:\n' + ov.to_string())
            print('pairs:\n' + n.to_string())
        d.to_json(RESULTS / f'{run}_calls.jsonl', orient='records', lines=True, force_ascii=False)
        rates.to_csv(RESULTS / f'{run}_rates.csv')
        if len(o):
            o.to_csv(RESULTS / f'{run}_overlap.csv', index=False)


if __name__ == '__main__':
    main()
