"""Extension analyses for the paper (called by make_paper_assets.py; each returns LaTeX number macros).

  coverage   monitoring from one US city: share of the businesses recommended in city B that are also
             recommended in city A, vs. in a repeat in B (ChatGPT, declared, E1 question as typed)
  distance   localization index of cited domains between US cities by distance (no gradient = categorical)
  topics     localization index by topic (figure paper/figures/topics.pdf)
  larger     gpt-5.4 vs gpt-5.4-mini on the same 60 questions and 6 locations (run e1_large)
"""
import itertools
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_pilot import jaccard, load  # noqa: E402
from extract_recs import norm  # noqa: E402
from geoaudit.common import DATA, RESULTS, ROOT, SEED  # noqa: E402

LOC = pd.read_csv(DATA / 'locations.csv').set_index('loc_id')
Q = pd.read_csv(DATA / 'questions.csv').set_index('qid')
US = list(LOC.index[LOC.country == 'US'])
RNG = np.random.default_rng(SEED)
GROUPS = ['explicit', 'implicit', 'nonlocal']


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (LOC.at[a, 'lat'], LOC.at[a, 'lng'], LOC.at[b, 'lat'], LOC.at[b, 'lng']))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def boot(x, n=2000):
    x = np.asarray(x)
    b = [RNG.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def pair_gaps(d, col='domains'):
    """Per between-location pair (same repeat): the question's twin overlap minus the pair's overlap."""
    rows = []
    for qid, g in d.groupby('qid'):
        recs = g.to_dict('records')
        tw = [jaccard(x[col], y[col]) for x, y in itertools.combinations(recs, 2) if x['loc_id'] == y['loc_id']]
        if not tw:
            continue
        t = np.nanmean(tw)
        for x, y in itertools.combinations(recs, 2):
            if x['loc_id'] != y['loc_id'] and x['rep'] == y['rep']:
                rows.append(dict(qid=qid, group=x['group'], a=x['loc_id'], b=y['loc_id'],
                                 gap=t - jaccard(x[col], y[col])))
    return pd.DataFrame(rows)


def index_by_question(p):
    return p.groupby('qid').gap.mean()


def coverage():
    r = pd.read_json(RESULTS / 'recs_all.jsonl', lines=True)
    r = r[(r.run == 'e1_natural') & r.loc_id.isin(US)]
    r['names'] = [{norm(x['name']) for x in it if x['type'] in ('local', 'chain', 'product')} for it in r['items']]
    rows = []
    for qid, g in r.groupby('qid'):
        by = {(x.loc_id, x.rep): x.names for x in g.itertuples()}
        cities = sorted({k[0] for k in by})
        for b in cities:
            if not by.get((b, 1)):
                continue
            if (b, 2) in by:
                rows.append(dict(qid=qid, kind='twin', c=len(by[(b, 2)] & by[(b, 1)]) / len(by[(b, 1)])))
            for a in cities:
                if a != b and (a, 1) in by:
                    rows.append(dict(qid=qid, kind='other', c=len(by[(a, 1)] & by[(b, 1)]) / len(by[(b, 1)])))
    c = pd.DataFrame(rows)
    c['group'] = c.qid.map(Q.group)
    m = c.groupby(['group', 'kind']).c.mean() * 100
    names = r.explode('names').dropna(subset=['names'])
    names = names[names.qid.map(Q.group) == 'explicit']
    one_city = (names.groupby(['qid', 'names']).loc_id.nunique() == 1).mean() * 100
    return {'covTwinNearMe': f"{m[('explicit', 'twin')]:.0f}", 'covOtherNearMe': f"{m[('explicit', 'other')]:.0f}",
            'covTwinImplicit': f"{m[('implicit', 'twin')]:.0f}", 'covOtherImplicit': f"{m[('implicit', 'other')]:.0f}",
            'covOneCityNearMe': f'{one_city:.0f}', 'covUSCities': str(len(US))}


def coverage_curve(draws=100):
    """Share of all businesses recommended across the 15 US cities (both repeats) that a monitor captures
    with k randomly chosen cities; figure paper/figures/coverage.pdf."""
    r = pd.read_json(RESULTS / 'recs_all.jsonl', lines=True)
    r = r[(r.run == 'e1_natural') & r.loc_id.isin(US)]
    r['names'] = [{norm(x['name']) for x in it if x['type'] in ('local', 'chain', 'product')} for it in r['items']]
    rows = []
    for qid, g in r.groupby('qid'):
        by_city = {c: set().union(*g[g.loc_id == c].names) for c in US if (g.loc_id == c).any()}
        cities = list(by_city)
        total = set().union(*by_city.values())
        if len(total) < 3 or len(cities) < len(US):
            continue
        for k in range(1, len(US) + 1):
            got = [len(set().union(*(by_city[c] for c in RNG.choice(cities, k, replace=False)))) / len(total)
                   for _ in range(draws)]
            rows.append(dict(qid=qid, group=Q.group[qid], k=k, share=np.mean(got)))
    c = pd.DataFrame(rows)
    curve = c.groupby(['group', 'k'])['share'].mean().unstack(0) * 100
    curve.to_csv(RESULTS / 'coverage_curve.csv')
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for group, color, label in (('explicit', '#0072B2', 'Near me'), ('implicit', '#D55E00', 'Depends on place'),
                                ('nonlocal', '#999999', 'Not local (no location effect)')):
        ax.plot(curve.index, curve[group], color=color, lw=1.6, marker='o', ms=3, label=label)
    ax.set_xlabel('US cities monitored', fontsize=7)
    ax.set_ylabel('Businesses captured (%)', fontsize=7)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#dddddd', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=False, loc='lower right')
    fig.savefig(ROOT / 'paper' / 'figures' / 'coverage.pdf', bbox_inches='tight')
    fig.savefig(RESULTS / 'coverage.png', dpi=200, bbox_inches='tight')
    n = c.groupby('group').qid.nunique()
    out = {'curveQuestionsNearMe': str(n['explicit'])}
    WORDS = {1: 'One', 5: 'Five', 10: 'Ten'}         # LaTeX macro names cannot contain digits
    for k in (1, 5, 10):
        for g in GROUPS:
            out[f'curve{WORDS[k]}{g.capitalize()}'] = f'{curve.at[k, g]:.0f}'
    return out


def distance(d):
    p = pair_gaps(d[d.loc_id.isin(US)])
    p['km'] = [km(a, b) for a, b in zip(p.a, p.b)]
    near = p[p.group == 'explicit']
    close, far = near[near.km < 1000].gap.mean(), near[near.km > 3000].gap.mean()
    rho = spearmanr(near.km, near.gap).statistic
    return {'distNearMeClose': f'{close:.2f}', 'distNearMeFar': f'{far:.2f}', 'distRho': f'{rho:.2f}'.replace('-', '$-$'),
            'distMinKm': f'{p.km.min():.0f}'}, p


def topics(p):
    """Figure: localization index between US cities by topic, near-me and place-dependent questions."""
    p = p.assign(topic=p.qid.map(Q.topic))
    rows = []
    for (topic, group), g in p[p.group.isin(['explicit', 'implicit'])].groupby(['topic', 'group']):
        m, lo, hi = boot(index_by_question(g))
        rows.append(dict(topic=topic, group=group, m=m, lo=lo, hi=hi))
    t = pd.DataFrame(rows)
    order = t[t.group == 'implicit'].sort_values('m').topic.tolist()
    short = {'legal, government and education': 'public services', 'telecom and utilities': 'telecom, utilities',
             'travel and leisure': 'travel, leisure', 'cars and transport': 'cars, transport',
             'food and dining': 'food, dining'}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    y = np.arange(len(order))
    for off, group, color, label in ((0.15, 'explicit', '#0072B2', 'Near me'), (-0.15, 'implicit', '#D55E00', 'Depends on place')):
        s = t[t.group == group].set_index('topic').loc[order]
        ax.errorbar(s.m, y + off, xerr=[s.m - s.lo, s.hi - s.m], fmt='o', ms=4, lw=1, color=color, label=label)
    ax.axvline(0, color='#555555', lw=0.8)
    ax.set_yticks(y, [short.get(x, x) for x in order], fontsize=7)
    ax.set_xlabel('Localization index, two US cities', fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#dddddd', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=False, loc='lower right')
    fig.savefig(ROOT / 'paper' / 'figures' / 'topics.pdf', bbox_inches='tight')
    fig.savefig(RESULTS / 'topics.png', dpi=200, bbox_inches='tight')
    t.to_csv(RESULTS / 'topic_index.csv', index=False)
    imp = t[t.group == 'implicit'].set_index('topic').m
    return {'topicTopImplicit': f"{imp.max():.2f}", 'topicTopName': short.get(imp.idxmax(), imp.idxmax()),
            'topicLowImplicit': f"{imp.min():.2f}".replace('-', '$-$'), 'topicLowName': short.get(imp.idxmin(), imp.idxmin())}


def larger():
    """gpt-5.4 (e1_large) vs gpt-5.4-mini (e1_natural) on the same questions and locations."""
    big = load('e1_large')
    if len(big) < 720:                          # run incomplete: visible placeholders
        print(f'e1_large has {len(big)} of 720 answers - larger-model macros are placeholders')
        keys = ['largeQuestions', 'largeLocations', 'largeAnswers'] + [
            f'{t}{k}{g.capitalize()}{ci}' for t in ('large', 'small') for g in GROUPS
            for k, ci in (('Idx', ''), ('Idx', 'CI'), ('Searches', ''), ('PlaceInQueries', ''))]
        return {k: r'\pending{?}' for k in keys}
    small = load('e1_natural')
    small = small[small.qid.isin(set(big.qid)) & small.loc_id.isin(set(big.loc_id))]
    out = {'largeQuestions': str(big.qid.nunique()), 'largeLocations': str(big.loc_id.nunique()),
           'largeAnswers': f'{len(big):,}'}
    for tag, d in (('large', big), ('small', small)):
        p = pair_gaps(d[d.cites])
        for g in GROUPS:
            m, lo, hi = boot(index_by_question(p[p.group == g]))
            G = g.capitalize()
            out[f'{tag}Idx{G}'] = f'{m:.2f}'.replace('-0.00', '0.00').replace('-', '$-$')
            out[f'{tag}Idx{G}CI'] = f'{lo:.2f} to {hi:.2f}'.replace('-', '$-$')
            s = d[d.group == g]
            out[f'{tag}Searches{G}'] = f'{s.searched.mean() * 100:.0f}'
            out[f'{tag}PlaceInQueries{G}'] = f'{s[s.searched].loc_in_query.mean() * 100:.0f}'
    return out


STATES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida',
          'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
          'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
          'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
          'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
          'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
STATE_RE = __import__('re').compile(r'\b(' + '|'.join(STATES) + r')\b')


def default_location():
    """ChatGPT without a declared location (E1 'none'): which place does it assume?"""
    out = {}
    nat = load('e1_natural')
    srch = load('e1_search')
    for tag, d in (('Nat', nat), ('Srch', srch)):
        near = d[d.group == 'explicit']
        out[f'defAsks{tag}None'] = f"{near[near.loc_id == 'none'].asks_location.mean() * 100:.0f}"
        out[f'defAsks{tag}Loc'] = f"{near[near.loc_id.isin(LOC.index)].asks_location.mean() * 100:.0f}"
    # similarity of 'none' answers to US cities vs other countries, and the closest city per question
    s = srch[srch.cites]
    rows = []
    for qid, g in s.groupby('qid'):
        none = {r.rep: r.domains for r in g[g.loc_id == 'none'].itertuples()}
        if len(none) < 2:
            continue
        twin = jaccard(none[1], none[2])
        for r in g[g.loc_id.isin(LOC.index)].itertuples():
            if r.rep in none:
                rows.append(dict(qid=qid, group=r.group, loc=r.loc_id, us=LOC.at[r.loc_id, 'country'] == 'US',
                                 j=jaccard(none[r.rep], r.domains), twin=twin))
    p = pd.DataFrame(rows)
    local = p[p.group != 'nonlocal']
    m = local.groupby('us').j.mean()
    out['defJUS'] = f'{m[True]:.2f}'
    out['defJOther'] = f'{m[False]:.2f}'
    out['defJTwin'] = f"{local.groupby('qid').twin.first().mean():.2f}"
    closest = local[local.us].groupby(['qid', 'loc']).j.mean().reset_index()
    closest = closest.loc[closest.groupby('qid').j.idxmax()]
    share = closest['loc'].value_counts(normalize=True) * 100
    out['defTopCity'] = LOC.at[share.index[0], 'city']
    out['defTopCityShare'] = f'{share.iloc[0]:.0f}'
    out['defUniformShare'] = f'{100 / len(US):.0f}'
    # places written into the queries of 'none' answers (near-me, searched)
    nn = srch[(srch.loc_id == 'none') & (srch.group == 'explicit') & srch.searched]
    raw = {}
    for f in (ROOT / 'data' / 'raw' / 'openai' / 'e1_search').glob('exp_*__none__*.json'):
        j = __import__('json').loads(f.read_text(encoding='utf-8'))
        raw[f.name] = ' '.join(j['queries'])
    states = [STATE_RE.findall(q) for q in raw.values()]
    out['defStateInQueries'] = f'{sum(bool(x) for x in states) / max(len(states), 1) * 100:.0f}'
    # do the answers name a US state or one of our cities? (near-me answers that cite something)
    names = [LOC.at[x, 'city'] for x in US]
    n = named = 0
    for f in (ROOT / 'data' / 'raw' / 'openai' / 'e1_search').glob('exp_*__none__*.json'):
        j = __import__('json').loads(f.read_text(encoding='utf-8'))
        if j['cited']:
            n += 1
            named += bool(STATE_RE.search(j['answer']) or any(c in j['answer'] for c in names))
    out['defAnswersNamePlace'] = f'{named / max(n, 1) * 100:.0f}'
    print('default location:', {k: out[k] for k in sorted(out)})
    return out


def search_arm():
    """Localization index of the "Search the web:" arm (all location pairs), from results/e1_index.csv."""
    e = pd.read_csv(RESULTS / 'e1_index.csv')
    e = e[(e.arm == 'e1_search') & (e.pairs == 'all locations')].set_index('group')
    return {f'searchArmIdx{g.capitalize()}': f"{e.at[g, 'index']:.2f}" for g in GROUPS}


def all_macros():
    m = search_arm()
    m.update(default_location())
    d = load('e1_natural')
    d = d[d.cites]
    m.update(coverage())
    m.update(coverage_curve())
    dm, p = distance(d)
    m.update(dm)
    m.update(topics(p))
    m.update(larger())
    return m


if __name__ == '__main__':
    for k, v in sorted(all_macros().items()):
        print(f'{k} = {v}')
