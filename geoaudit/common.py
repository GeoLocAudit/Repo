"""Shared paths and URL helpers. Run scripts from the project root, e.g. `python scripts/run.py`."""
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW = DATA / 'raw'
RESULTS = ROOT / 'results'
SEED = 42

try:                                        # API keys from ROOT/.env (see .env.example)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    pass

REDIRECT_HOSTS = ('vertexaisearch.cloud.google.com',)


def clean_url(u):
    """Lower-case host, drop www., utm_* parameters, fragment and trailing slash."""
    p = urlparse(u.strip())
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith('utm_')]
    host = p.netloc.lower().removeprefix('www.')
    return urlunparse((p.scheme.lower(), host, p.path.rstrip('/'), '', urlencode(q), ''))


def domain_of(u):
    """Host name without www."""
    return urlparse(u).netloc.lower().removeprefix('www.')


def is_redirect(u):
    return domain_of(u) in REDIRECT_HOSTS


def env(name, default=None):
    return os.environ.get(name, default)
