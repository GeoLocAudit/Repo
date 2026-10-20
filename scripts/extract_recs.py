"""Recommendations measure: which businesses, brands and products does an answer recommend?

Sample: 50 questions (17 explicit, 17 implicit, 16 non-local; seed 42), complete blocks:
  E1  e1_natural, all 20 location settings x 2 twins           (2,000 answers)
  E2  net_* sessions x 2 twins, Gemini and ChatGPT              (1,000 answers for five sessions)
A language model (gpt-5.4-mini, no tools) lists every business, brand or product the answer
recommends or names as an option, with a type:
  local     an individual local business (one or a few locations)
  chain     a multi-location company, national provider or retailer, or service brand
  product   a specific product, model or product line
  platform  a directory, review site, app or search tool used to find options (Yelp, Google Maps)
  public    a government body, public service or non-profit
Publishers cited only as sources are excluded. One cached JSON per answer in data/recs/ (git-ignored);
the combined table goes to results/recs_sample.jsonl.

  python scripts/extract_recs.py extract          # the 50-question sample
  python scripts/extract_recs.py extract all      # every E1 and E2 answer (results/recs_all.jsonl)
  python scripts/extract_recs.py check      # 100 random extractions for hand checking
  python scripts/extract_recs.py analyse
"""
import itertools
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geoaudit.common import DATA, RAW, RESULTS, SEED  # noqa: E402

MODEL = 'gpt-5.4-mini'
CACHE = DATA / 'recs'
TYPES = ('local', 'chain', 'product', 'platform', 'public')
PROMPT = """Below is an answer that an AI assistant gave to the question "{q}".

List every business, brand, company, product or service that the answer recommends or names as an option for the user.
Do not list websites or publications that are only cited as sources of information, and do not list generic categories ("a local plumber").
Give each name exactly as written in the answer, and a type:
- "local": an individual local business with one or a few locations
- "chain": a multi-location company, national or international provider, retailer or service brand
- "product": a specific product, model or product line
- "platform": a directory, review site, app or search tool used to find options (e.g. Yelp, Google Maps)
- "public": a government body, public service or non-profit organisation
Return only JSON: {{"items": [{{"name": "...", "type": "..."}}]}} (an empty list if there are none).

ANSWER:
{a}"""


def sample_questions(n=50):
    q = pd.read_csv(DATA / 'questions.csv')
    sizes = {'explicit': 17, 'implicit': 17, 'nonlocal': 16}
    return pd.concat([g.sample(sizes[k], random_state=SEED) for k, g in q.groupby('group')])


def sample_files(everything=False):
    """The 50-question sample, or (everything=True) all E1 answers (both arms) and all E2 sessions."""
    qids = set(sample_questions().qid)
    keep = (lambda f: True) if everything else (lambda f: f.name.split('__')[0] in qids)
    runs = ['e1_natural', 'e1_search'] if everything else ['e1_natural']
    files = [f for r in runs for f in (RAW / 'openai' / r).glob('*.json') if keep(f)]
    for engine in ('gemini', 'openai'):
        for d in sorted((RAW / engine).glob('net_*')):
            files += [f for f in d.glob('*.json') if keep(f)]
    return files


def norm(name):
    s = name.lower().replace('&', ' and ')
    s = re.sub(r'\b(inc|llc|ltd|limited|co|corp|corporation|company|plc|pty)\b\.?', ' ', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def extract_one(f):
    engine, run = f.parent.parent.name, f.parent.name
    out = CACHE / engine / run / f.name
    if out.exists():
        return json.loads(out.read_text(encoding='utf-8'))
    from openai import OpenAI
    j = json.loads(f.read_text(encoding='utf-8'))
    answer = j['answer'][:12000]
    items = []
    if answer.strip():
        r = OpenAI().responses.create(model=MODEL, input=PROMPT.format(q=j['question'], a=answer))
        text = r.output_text
        m = re.search(r'\{.*\}', text, re.S)
        items = json.loads(m.group(0)).get('items', []) if m else []
        items = [dict(name=str(x.get('name', '')).strip(), type=x.get('type') if x.get('type') in TYPES else 'other')
                 for x in items if str(x.get('name', '')).strip()]
    rec = dict(engine=engine, run=run, qid=j['qid'], loc_id=j['loc_id'], rep=j['rep'], items=items)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    return rec


def extract(everything=False):
    files = sample_files(everything)
    print(f'{len(files)} answers to extract (cached ones are skipped)')
    done, fails, lock = [0], [0], threading.Lock()

    def one(f):
        try:
            extract_one(f)
        except Exception as e:
            with lock:
                fails[0] += 1
            print(f'  {f.name}: {type(e).__name__}: {str(e)[:120]}')
            return
        with lock:
            done[0] += 1
            if done[0] % 250 == 0:
                print(f'  {done[0]}/{len(files)}')

    with ThreadPoolExecutor(10) as ex:
        list(ex.map(one, files))
    wanted = {(f.parent.parent.name, f.parent.name, f.name) for f in files}
    recs = [json.loads(p.read_text(encoding='utf-8')) for p in CACHE.glob('*/*/*.json')
            if (p.parent.parent.name, p.parent.name, p.name) in wanted]
    name = 'recs_all.jsonl' if everything else 'recs_sample.jsonl'
    pd.DataFrame(recs).to_json(RESULTS / name, orient='records', lines=True, force_ascii=False)
    print(f'done {done[0]}, failed {fails[0]}; {len(recs)} records -> results/{name}')


def check():
    if (RESULTS / 'recs_check_sample.csv').exists():
        sys.exit('results/recs_check_sample.csv exists and holds hand checks - not overwritten.')
    d = pd.read_json(RESULTS / 'recs_sample.jsonl', lines=True)
    s = d.sample(100, random_state=SEED)
    rows = []
    for r in s.itertuples():
        j = json.loads((RAW / r.engine / r.run / f'{r.qid}__{r.loc_id}__r{r.rep}.json').read_text(encoding='utf-8'))
        rows.append(dict(engine=r.engine, run=r.run, qid=r.qid, loc_id=r.loc_id, rep=r.rep, question=j['question'],
                         answer=j['answer'][:3000], items='; '.join(f"{x['name']} [{x['type']}]" for x in r.items),
                         missing='', wrong='', type_errors='', ok=''))
    pd.DataFrame(rows).to_csv(RESULTS / 'recs_check_sample.csv', index=False)
    print('100 extractions -> results/recs_check_sample.csv (fill missing / wrong / type_errors / ok)')


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def analyse():
    rng = np.random.default_rng(SEED)
    src = RESULTS / ('recs_all.jsonl' if (RESULTS / 'recs_all.jsonl').exists() else 'recs_sample.jsonl')
    d = pd.read_json(src, lines=True)
    d = d[d.run != 'net_ca2']                   # the Canada repeat measures drift, not location
    print(f'{src.name}: {len(d)} answers')
    q = pd.read_csv(DATA / 'questions.csv').set_index('qid')
    loc = pd.read_csv(DATA / 'locations.csv').set_index('loc_id')
    d['group'] = d.qid.map(q.group)
    d['names'] = [sorted({norm(x['name']) for x in it if x['type'] in ('local', 'chain', 'product') and norm(x['name'])}) for it in d['items']]
    d['n_local'] = [sum(x['type'] == 'local' for x in it) for it in d['items']]
    d['n_chain'] = [sum(x['type'] == 'chain' for x in it) for it in d['items']]
    e1 = d.run.str.startswith('e1_')
    d['place'] = np.where(e1, d.loc_id, d.run)
    rows = []
    for (engine, exp), g in d.groupby(['engine', np.where(e1, d.run.str.replace('e1_', 'E1 '), 'E2')]):
        if exp.startswith('E1'):
            g = g[g.loc_id != 'none']
        for group, gg in g.groupby('group'):
            diffs = []
            for qid, h in gg[gg.names.str.len() > 0].groupby('qid'):
                recs = h.to_dict('records')
                tw, bt = [], []
                for x, y in itertools.combinations(recs, 2):
                    if x['place'] == y['place']:
                        tw.append(jacc(x['names'], y['names']))
                    elif x['rep'] == y['rep']:
                        bt.append(jacc(x['names'], y['names']))
                if tw and bt:
                    diffs.append((np.nanmean(tw), np.nanmean(bt)))
            if not diffs:
                continue
            a = np.array(diffs)
            idx = a[:, 0] - a[:, 1]
            b = [rng.choice(idx, len(idx)).mean() for _ in range(2000)]
            rows.append(dict(experiment=exp, engine=engine, group=group, twin=a[:, 0].mean(), between=a[:, 1].mean(),
                             index=idx.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), questions=len(idx),
                             names_per_answer=gg.names.str.len().mean(),
                             answers_naming_local=(gg.n_local > 0).mean() * 100,
                             answers_naming_chain=(gg.n_chain > 0).mean() * 100))
    r = pd.DataFrame(rows)
    pd.set_option('display.width', 180)
    print('Recommended names (local businesses, chains, products): overlap twin vs between places, index = twin - between\n' +
          r.round(3).to_string(index=False))
    e1 = d[(d.run == 'e1_natural') & d.loc_id.isin(loc.index) & (d.group == 'explicit')]
    by = e1.groupby('loc_id').agg(local=('n_local', lambda s: (s > 0).mean() * 100),
                                  chain=('n_chain', lambda s: (s > 0).mean() * 100), answers=('qid', 'size'))
    by['city'] = loc.city
    print('\nE1 explicit questions: answers naming a local business / a chain, % by location\n' +
          by.round(0).sort_values('local').to_string())
    r.to_csv(RESULTS / 'recs_index.csv', index=False)
    by.to_csv(RESULTS / 'recs_local_by_location.csv')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'extract'
    if cmd == 'extract':
        extract(everything='all' in sys.argv[2:])
    else:
        {'check': check, 'analyse': analyse}[cmd]()
