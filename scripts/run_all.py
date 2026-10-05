"""Run each independent variant in a fresh process; never overwrite a run."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--profile',choices=('smoke','formal'),default='smoke')
p.add_argument('--device',default='cpu')
p.add_argument('--dtype',choices=('float32','bfloat16'),default='float32')
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
variants=('baseline','rmsnorm','swiglu','nope','rope','gqa')
if any((a.out/v).exists() for v in variants):
    p.error('Run directories already exist; use a new output directory to retain all records.')
for v in variants:
    subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),'--variant',v,'--profile',a.profile,
                    '--device',a.device,'--dtype',a.dtype,'--out',str(a.out/v)],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts'/'plot_experiments.py'),
                *[str(a.out/v) for v in variants],'--out',str(a.out/'comparison')],check=True)
