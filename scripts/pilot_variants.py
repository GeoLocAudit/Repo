"""Pilot: which ChatGPT set-up searches for implicit and non-local questions?

Variants on the 30 pilot questions at Boise and London, one repeat each:
  mini     gpt-5.4-mini, question as typed
  prefix   gpt-5.4-nano, "Search the web: {question}"
Output: data/raw/openai/variant_<name>/<qid>__<loc>__r1.json, then a rate table per group.

  python scripts/pilot_variants.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoaudit.common import DATA, RAW  # noqa: E402
from geoaudit.engines import _parse_openai, openai_location  # noqa: E402

VARIANTS = {'mini': ('gpt-5.4-mini', '{q}'), 'prefix': ('gpt-5.4-nano', 'Search the web: {q}'),
            'mini_prefix': ('gpt-5.4-mini', 'Search the web: {q}')}


def call(model, text, loc):
    from openai import OpenAI
    tool = {'type': 'web_search', 'user_location': openai_location(loc)}
    r = OpenAI().responses.create(model=model, input=text, tools=[tool], include=['web_search_call.action.sources'])
    return _parse_openai(r.model_dump())


def main():
    q = pd.read_csv(DATA / 'questions.csv').query('pilot == 1')
    loc = pd.read_csv(DATA / 'locations.csv').set_index('loc_id').loc[['boi', 'lon']]
    rows = []
    for name, (model, tmpl) in VARIANTS.items():
        out = RAW / 'openai' / f'variant_{name}'
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(r, lid, l) for r in q.itertuples() for lid, l in loc.iterrows()]

        def one(job):
            r, lid, l = job
            f = out / f'{r.qid}__{lid}__r1.json'
            if f.exists():
                res = json.loads(f.read_text(encoding='utf-8'))
            else:
                res = call(model, tmpl.format(q=r.question), l.to_dict())
                res.update(qid=r.qid, loc_id=lid, model=model, prompt=tmpl.format(q=r.question))
                f.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
            city = ' '.join(res['queries']).lower()
            return dict(variant=name, group=r.group, loc_id=lid, searched=bool(res['queries']),
                        cites=bool(res['cited']), loc_in_query=l.city.lower() in city or l.region.lower() in city)

        with ThreadPoolExecutor(6) as ex:
            rows += list(ex.map(one, jobs))
    d = pd.DataFrame(rows)
    print(d.groupby(['variant', 'group'])[['searched', 'cites', 'loc_in_query']].mean().mul(100).round(0).to_string())


if __name__ == '__main__':
    main()
