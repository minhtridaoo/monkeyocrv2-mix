"""Locate Tuấn's MONKEY_V2 checkout (pipeline reference) and record which commit was used.

Resolution order: environment variable MONKEY_V2_ROOT, then the sibling checkout
`<this repo>/../../MONKEY_V2` used on the member's machine (`Monkey v2/MONKEY_V2`).
"""
import os
import subprocess
import sys
from pathlib import Path

from mixab import require

WORKSPACE = Path(__file__).resolve().parents[1]


def monkey_root(required=True):
    candidates = [os.environ.get('MONKEY_V2_ROOT'), WORKSPACE.parents[1]/'MONKEY_V2']
    for candidate in candidates:
        if candidate and (Path(candidate)/'rev4').is_dir() and (Path(candidate)/'lowres').is_dir():
            return Path(candidate).resolve()
    require(not required, 'MONKEY_V2 checkout not found; set MONKEY_V2_ROOT')
    return None


def use_monkey():
    """Put MONKEY_V2 on sys.path so `rev4` / `lowres` can be imported; return its identity."""
    root = monkey_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return identity(root)


def identity(root=None):
    root = root or monkey_root()
    def git(*args):
        try:
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except OSError:
            return ''
    return dict(path=str(root), commit=git('rev-parse', 'HEAD'),
                dirty=bool(git('status', '--porcelain', '--untracked-files=no')))
