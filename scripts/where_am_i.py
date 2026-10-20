"""Check which location the engines infer from this machine's network (run before each VPN session).

Sends 5 "near me" questions to Gemini (no declared location) and prints the places it writes into
its searches; ChatGPT gets one question as a control. About 6 calls.

  python scripts/where_am_i.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoaudit.engines import ask_gemini, ask_openai  # noqa: E402

QUESTIONS = ['best plumber near me', 'urgent care near me open now', 'banks near me with no monthly fees',
             'cheapest gas near me', 'best internet providers in my area']
PLACE = re.compile(r'\b(in|near|around)\s+([A-Z][\w.]*(?:[ ,]+[A-Z][\w.]*){0,3})')


def main():
    places = Counter()
    for q in QUESTIONS:
        r = ask_gemini(q)
        found = [m.group(2) for x in r['queries'] for m in PLACE.finditer(x)]
        found += [w for x in r['queries'] for w in ('Canada', 'United States', 'USA', 'US', 'UK', 'United Kingdom', 'Australia', 'AU') if re.search(rf'\b{w}\b', x)]
        places.update(found)
        print(f'gemini | {q:38} | {r["queries"][:3]}')
    r = ask_openai(QUESTIONS[-1])
    print(f'chatgpt| {QUESTIONS[-1]:38} | {r["queries"][:3]} | {r["answer"][:120]!r}')
    print('\nplaces in Gemini queries:', dict(places.most_common(5)) or 'none')


if __name__ == '__main__':
    main()
