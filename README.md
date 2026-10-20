# GeoLocAudit

Does the user's location change which sources generative search engines cite?

A controlled audit: the same questions are sent to ChatGPT (gpt-5.4-nano, `web_search` with
`user_location`) and Gemini (gemini-3.1-flash-lite, Google Search grounding with `latLng`) from
different locations, each twice, so that location effects can be separated from run-to-run noise.

## Layout

| Path | Content |
|---|---|
| `data/questions.csv` | 300 questions: 10 topics x 3 groups (explicitly local, implicitly local, not local); no question names a place |
| `data/locations.csv` | 15 US cities and 4 other countries, with timezone and coordinates; `pilot` marks the pilot set |
| `data/raw/` | one JSON per API call (not in git) |
| `geoaudit/` | shared code: URL helpers, engine calls |
| `scripts/make_questions.py` | writes `data/questions.csv` |
| `scripts/run.py` | runs questions x locations x repeats against one engine |
| `scripts/analyse_pilot.py` | search / clarification rates, location in queries, citation overlap |
| `decisions.md` | design decisions, dated |

## Setup

Python 3.14; `pip install -r requirements.txt`; API keys in `.env` (see `.env.example`).

```
python scripts/make_questions.py
python scripts/run.py --engine openai --run pilot1 --pilot --reps 2
python scripts/analyse_pilot.py pilot1
```

Gemini calls share a daily grounding quota with another project and are refused unless
`--allow-gemini` is passed.
