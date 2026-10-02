"""Explicit external BFM source binding; never imports another BodyISA checkout."""
from pathlib import Path
import os

_root = None


def configure_external(path):
    global _root
    candidate = Path(path).resolve()
    if _root is not None and _root != candidate:
        raise RuntimeError('BFM_SOURCE_CHANGE_REQUIRES_NEW_PROCESS')
    for relative in ('env.py', 'common.py', 'bfm_zero_inference_code/fb_cpr_aux/model.py'):
        if not (candidate / relative).is_file():
            raise FileNotFoundError(f'EXTERNAL_BFM_SOURCE_MISSING: {candidate / relative}')
    _root = candidate
    return _root


def external_root():
    if _root is not None:
        return _root
    path = os.environ.get('BODYISA_BFM_ROOT')
    if not path:
        raise RuntimeError('BFM_ROOT_REQUIRED: configure bfm_root or BODYISA_BFM_ROOT')
    return configure_external(path)
