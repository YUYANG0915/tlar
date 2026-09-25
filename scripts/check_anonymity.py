#!/usr/bin/env python3
"""Inspect a source release for credentials, local identities and private artifacts."""
import argparse
import json
from pathlib import Path
import re

SKIP={'__pycache__','.pytest_cache','.venv'}
PATTERNS={
    'user_home':r'/(?:Users|home)/[A-Za-z0-9_.-]+',
    'email':r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'hf_credential':r'\bhf_[A-Za-z0-9]{20,}\b',
    'api_credential':r'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b',
    'github_credential':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
    'cloud_key':r'\bAKIA[A-Z0-9]{16}\b',
    'institution_cluster_path':r'/(?:scratch|work|orange|ufrc)/[A-Za-z0-9_.-]+',
    'private_remote':r'git@[^\s]+',
}


def inspect(root,extra_terms=()):
    findings=[];count=0
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if any(p in SKIP for p in rel.parts):continue
        if path.is_symlink():findings.append({'file':str(rel),'category':'symlink'});continue
        if path.is_dir():
            if path.name=='.git':findings.append({'file':str(rel),'category':'git_metadata'})
            continue
        count+=1
        if path.name in ('.DS_Store','.env') or path.suffix in ('.pem','.key','.pyc','.pkl','.pt','.bin'):
            findings.append({'file':str(rel),'category':'private_or_binary_artifact'})
        try:text=path.read_text()
        except UnicodeDecodeError:
            findings.append({'file':str(rel),'category':'binary_requires_review'});continue
        # Pattern definitions contain literal marker text by construction.
        checks={} if path.name=='check_anonymity.py' else PATTERNS
        for name,pattern in checks.items():
            if re.search(pattern,text):findings.append({'file':str(rel),'category':name})
        for term in extra_terms:
            if term and term.casefold() in (str(rel)+'\n'+text).casefold():
                findings.append({'file':str(rel),'category':'private_term'})
        if path.suffix=='.json':
            def walk(obj):
                if isinstance(obj,dict):
                    for k,v in obj.items():
                        if re.fullmatch(r'(?:api[_-]?key|access[_-]?token|password|secret|hf[_-]?token)',k,re.I) and v:
                            findings.append({'file':str(rel),'category':'populated_credential_field'})
                        walk(v)
                elif isinstance(obj,list):
                    for v in obj:walk(v)
            walk(json.loads(text))
    return {'files_scanned':count,'findings':findings,'passed':not findings}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--private-terms',type=Path,help='External JSON list of names, affiliations, usernames and paths')
    a=p.parse_args();terms=json.loads(a.private_terms.read_text()) if a.private_terms else []
    result=inspect(a.root,terms);print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
