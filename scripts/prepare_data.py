"""Download with HTTP validation, then reproduce the upstream character split."""
from pathlib import Path
import hashlib
import json
import runpy
import requests
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data'/'shakespeare_char'/'input.txt'
url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not p.exists():
    r=requests.get(url,timeout=60)
    r.raise_for_status()
    text=r.content.decode('utf-8')
    if len(text)!=1115394 or len(set(text))!=65 or not text.startswith('First Citizen:'):
        raise ValueError('Unexpected dataset; refusing to encode response.')
    p.write_bytes(r.content)
text=p.read_text()
if len(text)!=1115394 or len(set(text))!=65: raise ValueError('Unexpected existing input.txt')
runpy.run_path(str(p.with_name('prepare.py')),run_name='__main__')
meta={'url':url,'characters':len(text),'vocab_size':len(set(text)),
      'split':'first 90% train; remaining 10% validation',
      'sha256':{name:hashlib.sha256(p.with_name(name).read_bytes()).hexdigest() for name in ('input.txt','train.bin','val.bin')}}
p.with_name('manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
