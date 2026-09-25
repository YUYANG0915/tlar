"""Validate the packaged sources and run local correctness tests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',action='store_true',help='Include local tiny-model execution tests')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'docs/source_manifest.json').read_text())
    for row in manifest['files']:
        if hashlib.sha256((root/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Source digest mismatch: '+row['path'])
    for path in root.rglob('*.py'):
        if '.venv' not in path.parts:ast.parse(path.read_text(),filename=str(path))
    tests=['tests/test_tlar_adaptive_tree.py','tests/test_section4_causality.py',
           'tests/test_causal_suite.py','tests/test_submit_hf_tree.py',
           'tests/test_adaptive_controller.py','tests/test_analysis_regressions.py',
           'tests/test_paper_protocol.py']
    if a.models:tests+=['tests/test_tlar_hf_tree.py','tests/test_paper_execution.py']
    subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=root,check=True)
    subprocess.run([sys.executable,'scripts/check_anonymity.py'],cwd=root,check=True)
    print('PASS: source hashes, Python syntax and selected correctness tests')

if __name__=='__main__':main()
