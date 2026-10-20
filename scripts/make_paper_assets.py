"""Paper assets from the results: figure, number macros (paper/numbers.tex), table rows (paper/table_rows.tex).

Reads results/e1_index.csv, e1_place_sources.csv, e1_natural_rates.csv (E1) and the raw E2 sessions.
E2 indices are computed here by pair type: different countries vs the same country (US East vs US West),
and the Canada repeat (drift).

  python scripts/make_paper_assets.py
"""
import itertools
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_pilot import jaccard, load  # noqa: E402
from geoaudit.common import RESULTS, ROOT, SEED  # noqa: E402

PAPER = ROOT / 'paper'
COUNTRY = {'net_ca1': 'CA', 'net_ca2': 'CA', 'net_us_east': 'US', 'net_us_west': 'US', 'net_uk': 'GB', 'net_au': 'AU'}
WORD = {'CA': r'\bCanad', 'US': r'United States|\bUSA\b|\bU\.S\.', 'GB': r'\bUK\b|United Kingdom|\bBritain', 'AU': r'Australia'}
GROUPS = ['explicit', 'implicit', 'nonlocal']
LABEL = {'explicit': 'Near me', 'implicit': 'Depends on place', 'nonlocal': 'Not local'}
RNG = np.random.default_rng(SEED)


def boot(diff, n=2000):
    diff = np.asarray(diff)
    b = [RNG.choice(diff, len(diff)).mean() for _ in range(n)]
    return diff.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(diff)


def e2_pairs(engine, sessions):
    d = pd.concat([load(s, engine).assign(session=s) for s in sessions], ignore_index=True)
    rows = []
    for qid, g in d[d.cites].groupby('qid'):
        recs = g.to_dict('records')
        for x, y in itertools.combinations(recs, 2):
            if x['session'] == y['session']:
                kind = 'twin'
            elif x['rep'] != y['rep']:
                continue
            elif COUNTRY[x['session']] == COUNTRY[y['session']]:
                kind = 'same country'
            else:
                kind = 'different countries'
            rows.append(dict(qid=qid, group=x['group'], kind=kind, j=jaccard(x['domains'], y['domains'])))
    return d, pd.DataFrame(rows)


def e2_index(p, kind):
    out = {}
    for group in GROUPS:
        g = p[p.group == group]
        tw = g[g.kind == 'twin'].groupby('qid').j.mean()
        bt = g[g.kind == kind].groupby('qid').j.mean()
        out[group] = boot((tw - bt).dropna())
    return out


def rec_pairs(d, place, country):
    """Pairs of answers (same question) for the recommended names; only answers that name something."""
    rows = []
    d = d.assign(place=place, country=country)
    for qid, g in d[d.names.str.len() > 0].groupby('qid'):
        recs = g.to_dict('records')
        for x, y in itertools.combinations(recs, 2):
            if x['place'] == y['place']:
                kind = 'twin'
            elif x['rep'] != y['rep']:
                continue
            elif x['country'] == y['country']:
                kind = 'same country'
            else:
                kind = 'different countries'
            rows.append(dict(qid=qid, group=x['group'], kind=kind, j=jaccard(x['names'], y['names'])))
    return pd.DataFrame(rows)


def recommendation_macros():
    """Localization index of the businesses, brands and products the answers recommend (results/recs_all.jsonl)."""
    from extract_recs import norm
    loc = pd.read_csv(ROOT / 'data' / 'locations.csv').set_index('loc_id')
    group = pd.read_csv(ROOT / 'data' / 'questions.csv').set_index('qid').group
    d = pd.read_json(RESULTS / 'recs_all.jsonl', lines=True)
    d['group'] = d.qid.map(group)
    d['names'] = [sorted({norm(x['name']) for x in it if x['type'] in ('local', 'chain', 'product') and norm(x['name'])})
                  for it in d['items']]
    out = {}
    e1 = d[(d.run == 'e1_natural') & d.loc_id.isin(loc.index)]
    p = rec_pairs(e1, e1.loc_id, e1.loc_id.map(loc.country))
    for kind, tag in (('same country', 'recGptDeclCity'), ('different countries', 'recGptDeclCountry')):
        for g, (m, lo, hi, _) in e2_index(p, kind).items():
            out[f'{tag}{g.capitalize()}'] = num(m)
            out[f'{tag}{g.capitalize()}CI'] = ci(lo, hi)
    for engine, tag in (('gemini', 'recGemNetCountry'), ('openai', 'recGptNetCountry')):
        e2 = d[(d.engine == engine) & d.run.isin([s for s in COUNTRY if s != 'net_ca2'])]
        p = rec_pairs(e2, e2.run, e2.run.map(COUNTRY))
        for g, (m, lo, hi, _) in e2_index(p, 'different countries').items():
            out[f'{tag}{g.capitalize()}'] = num(m)
            out[f'{tag}{g.capitalize()}CI'] = ci(lo, hi)
    e1n = e1[e1.group == 'explicit']
    out['recNamesPerNearMe'] = f'{e1n.names.str.len().mean():.1f}'
    p1 = rec_pairs(e1n, e1n.loc_id, e1n.loc_id.map(loc.country))
    out['recTwinNearMe'] = f"{p1[p1.kind == 'twin'].groupby('qid').j.mean().mean() * 100:.0f}"
    out['recBetweenNearMe'] = f"{p1[p1.kind != 'twin'].groupby('qid').j.mean().mean() * 100:.0f}"
    return out


def num(x):
    """Two decimals for LaTeX: no negative zero, minus sign in math mode."""
    s = f'{x:.2f}'
    s = '0.00' if s == '-0.00' else s
    return s.replace('-', '$-$')


def ci(lo, hi):
    return f'{num(lo)} to {num(hi)}'


def main():
    PAPER.mkdir(exist_ok=True)
    (PAPER / 'figures').mkdir(exist_ok=True)
    five = ['net_ca1', 'net_us_east', 'net_us_west', 'net_uk', 'net_au']
    series, macros = {}, {}

    # E1 (declared location, ChatGPT, natural arm)
    e1 = pd.read_csv(RESULTS / 'e1_index.csv')
    e1 = e1[e1.arm == 'e1_natural']
    for key, pairs in (('two US cities', 'gptDeclCity'), ('US city vs other country', 'gptDeclCountry')):
        sub = e1[e1.pairs == key].set_index('group')
        series[key] = {g: (sub.at[g, 'index'], sub.at[g, 'lo'], sub.at[g, 'hi'], sub.at[g, 'questions']) for g in GROUPS}
        for g in GROUPS:
            macros[f'{pairs}{g.capitalize()}'] = num(sub.at[g, 'index'])

    # E2 (network location)
    for engine, tag in (('gemini', 'gem'), ('openai', 'gpt')):
        d, p = e2_pairs(engine, five)
        for kind, name in (('different countries', 'NetCountry'), ('same country', 'NetSameCountry')):
            idx = e2_index(p, kind)
            series[f'{engine} {kind}'] = idx
            for g in GROUPS:
                m, lo, hi, _ = idx[g]
                macros[f'{tag}{name}{g.capitalize()}'] = num(m)
                macros[f'{tag}{name}{g.capitalize()}CI'] = ci(lo, hi)
        _, pd2 = e2_pairs(engine, ['net_ca1', 'net_ca2'])
        drift = e2_index(pd2, 'same country')
        for g in GROUPS:
            macros[f'{tag}Drift{g.capitalize()}'] = num(drift[g][0])
        if engine == 'gemini':
            raw = pd.concat([pd.read_json(f, typ='series').to_frame().T.assign(session=s)
                             for s in five for f in (ROOT / 'data' / 'raw' / 'gemini' / s).glob('*.json')],
                             ignore_index=True)
            raw['group'] = raw.qid.str[:3].map({'exp': 'explicit', 'imp': 'implicit', 'non': 'nonlocal'})
            raw['named'] = [bool(re.search(WORD[COUNTRY[s]], ' '.join(q or []))) for s, q in zip(raw.session, raw.queries)]
            for g, v in raw.groupby('group').named.mean().mul(100).items():
                macros[f'gemCountryInQueries{g.capitalize()}'] = f'{v:.0f}'

    # E1 behaviour and place-specific sources
    rates = pd.read_csv(RESULTS / 'e1_natural_rates.csv').set_index('group')
    src = pd.read_csv(RESULTS / 'e1_place_sources.csv')
    src = src[src.arm == 'e1_natural'].set_index(['group', 'locations'])
    for g in GROUPS:
        G = g.capitalize()
        macros[f'gptSearches{G}'] = f"{rates.at[g, 'searched']:.0f}"
        macros[f'gptPlaceInQueries{G}'] = f"{rates.at[g, 'loc_in_query']:.0f}"
        macros[f'gptOwnPlaceUS{G}'] = f"{src.at[(g, 'US cities'), 'own_place']:.0f}"
        macros[f'gptOwnPlaceOther{G}'] = f"{src.at[(g, 'other countries'), 'own_place']:.0f}"
    ch = pd.read_csv(RESULTS / 'e1_channel.csv')
    ch = ch[(ch.arm == 'e1_natural') & (ch.group == 'implicit')].set_index('place_in_queries')
    macros['gptBackendImplicit'] = num(ch.at['neither', 'index'])
    macros['gptBackendImplicitCI'] = ci(ch.at['neither', 'lo'], ch.at['neither', 'hi'])
    macros['gptRewriteImplicit'] = num(ch.at['both', 'index'])

    # figure: localisation index by question group and comparison
    show = [('two US cities', 'ChatGPT declared: two US cities', '#0072B2', 'o'),
            ('US city vs other country', 'ChatGPT declared: two countries', '#56B4E9', 'o'),
            ('gemini different countries', 'Gemini network: two countries', '#D55E00', 's'),
            ('gemini same country', 'Gemini network: US East vs West', '#E69F00', 's'),
            ('openai different countries', 'ChatGPT network: two countries', '#999999', '^')]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    width = 0.14
    for i, (key, label, color, marker) in enumerate(show):
        x = np.arange(3) + (i - 2) * width
        m = [series[key][g][0] for g in GROUPS]
        lo = [series[key][g][0] - series[key][g][1] for g in GROUPS]
        hi = [series[key][g][2] - series[key][g][0] for g in GROUPS]
        ax.errorbar(x, m, yerr=[lo, hi], fmt=marker, color=color, ms=5, lw=1.2, capsize=0, label=label)
    ax.axhline(0, color='#555555', lw=0.8)
    ax.set_xticks(range(3), [LABEL[g] for g in GROUPS], fontsize=8)
    ax.set_ylabel('Localisation index', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', lw=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.32)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6, frameon=False, loc='lower center', ncol=2,
               handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.55, 0.0))
    fig.savefig(PAPER / 'figures' / 'localisation.pdf', bbox_inches='tight')
    fig.savefig(RESULTS / 'localisation.png', dpi=200, bbox_inches='tight')

    macros.update(recommendation_macros())
    from extensions import all_macros
    macros.update(all_macros())
    with open(PAPER / 'numbers.tex', 'w', encoding='utf-8') as f:
        f.write('%% generated by scripts/make_paper_assets.py - do not edit\n')
        for k, v in sorted(macros.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    print(f'{len(macros)} macros -> paper/numbers.tex; figure -> paper/figures/localisation.pdf')
    for k in sorted(macros):
        print(f'  {k} = {macros[k]}')


if __name__ == '__main__':
    main()
