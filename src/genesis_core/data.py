"""Packaged, source-bound historical inputs; no network or lab-path access."""
import json
from importlib.resources import files


def load(name):
    if name not in {'analysis', 'construction', 'entropy_laws', 'qualification'}:
        raise ValueError('unknown bundled dataset')
    return json.loads(files('genesis_core').joinpath('data', name + '.json').read_text(encoding='utf-8'))
