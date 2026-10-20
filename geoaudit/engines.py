"""One call to each engine's API with web search on, at a given user location.

Both return the same layout: queries, consulted, cited, answer, usage (plus the location sent).

  ChatGPT  gpt-5.4-nano, Responses API, web_search tool. Location goes in the tool's
           user_location (city, region, ISO country, IANA timezone). loc=None sends no
           user_location (the server-side default); loc={} sends {"type": "approximate"} only.
  Gemini   gemini-3.1-flash-lite, google_search grounding. Location goes in
           toolConfig.retrievalConfig.latLng; whether Search grounding honours it is what the
           pilot checks (it is documented for Maps grounding).
Keys are sent in headers only (never in URLs or logs).
"""
import threading
import time

import requests

from .common import env, is_redirect

OPENAI_MODEL = 'gpt-5.4-nano'
GEMINI_API = 'https://generativelanguage.googleapis.com/v1beta'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36'
_redirects, _rlock = {}, threading.Lock()


def gemini_model():
    return env('GEMINI_MODEL', 'gemini-3.1-flash-lite')


def resolve(u):
    """Follow one Vertex AI Search redirect link to the real URL (cached per run)."""
    if not is_redirect(u):
        return u
    with _rlock:
        if u in _redirects:
            return _redirects[u]
    real = None
    for _ in range(2):
        try:
            r = requests.get(u, headers={'User-Agent': UA}, timeout=20, allow_redirects=False)
            real = r.headers.get('Location')
            break
        except requests.RequestException:
            time.sleep(2)
    with _rlock:
        _redirects[u] = real or u
    return real or u


def openai_location(loc):
    if loc is None:
        return None
    out = {'type': 'approximate'}
    for k in ('city', 'region', 'country', 'timezone'):
        if loc.get(k):
            out[k] = loc[k]
    return out


FOLLOWUP = "I'd rather not share my exact address. Just give me your best recommendations."


def ask_openai(prompt, loc=None, followup=None, model=None):
    """One answer; if followup is given and the answer cites nothing (typically the model asked
    for a ZIP code instead of searching), send it as a second turn and record that turn too."""
    from openai import OpenAI
    client = OpenAI()
    model = model or OPENAI_MODEL
    tool = {'type': 'web_search'}
    ul = openai_location(loc)
    if ul is not None:
        tool['user_location'] = ul
    r = client.responses.create(model=model, input=prompt, tools=[tool],
                                include=['web_search_call.action.sources'])
    out = dict(model=model, location_sent=ul, **_parse_openai(r.model_dump()))
    if followup and not out['cited']:
        r2 = client.responses.create(model=model, previous_response_id=r.id, input=followup,
                                     tools=[tool], include=['web_search_call.action.sources'])
        out['followup'] = dict(text=followup, **_parse_openai(r2.model_dump()))
    return out


def _parse_openai(d):
    queries, consulted, opened, cited, calls, answer = [], [], [], [], [], ''
    for o in d['output']:
        if o['type'] == 'web_search_call':
            a = o.get('action') or {}
            if a.get('type') == 'search':
                q = a.get('queries') or ([a['query']] if a.get('query') else [])
                src = [s['url'] for s in a.get('sources') or [] if s.get('url')]
                queries += q
                consulted += src
                calls.append(dict(queries=q, sources=src))
            elif a.get('url'):
                opened.append(a['url'])
        elif o['type'] == 'message':
            for c in o['content']:
                answer += c.get('text') or ''
                cited += [x['url'] for x in c.get('annotations') or [] if x.get('type') == 'url_citation']
    return dict(queries=queries, consulted=consulted, opened=opened, cited=cited, calls=calls,
                answer=answer, usage=d.get('usage'))


def ask_gemini(prompt, loc=None, followup=None, model=None):
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'tools': [{'google_search': {}}]}
    sent = None
    if loc and loc.get('lat') is not None:
        sent = {'latitude': float(loc['lat']), 'longitude': float(loc['lng'])}
        body['toolConfig'] = {'retrievalConfig': {'latLng': sent}}
    for attempt in range(4):
        r = requests.post(f'{GEMINI_API}/models/{gemini_model()}:generateContent',
                          headers={'x-goog-api-key': env('GEMINI_API_KEY')}, json=body, timeout=120)
        if r.status_code not in (429, 500, 503):
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    d = r.json()
    cand = (d.get('candidates') or [{}])[0]
    g = cand.get('groundingMetadata') or {}
    chunks = [(c.get('web') or {}) for c in g.get('groundingChunks') or []]
    urls = [resolve(c.get('uri', '')) for c in chunks]
    used = sorted({i for s in g.get('groundingSupports') or [] for i in s.get('groundingChunkIndices') or []})
    answer = ''.join(p.get('text', '') for p in (cand.get('content') or {}).get('parts') or [])
    return dict(model=gemini_model(), location_sent=sent, queries=g.get('webSearchQueries') or [],
                consulted=urls, consulted_titles=[c.get('title') for c in chunks], opened=[],
                cited=[urls[i] for i in used if i < len(urls)], answer=answer, usage=d.get('usageMetadata'))


ASK = {'openai': ask_openai, 'gemini': ask_gemini}
