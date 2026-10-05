"""Single-device controlled HW1 runs. Formal runs always use 5,000 updates."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from importlib.metadata import distributions
import time

import numpy as np
import torch
from model_variants import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent
VARIANTS = ('baseline', 'rmsnorm', 'swiglu', 'nope', 'rope', 'gqa')


def batch(data, size, length, rng, device):
    starts = torch.randint(len(data)-length, (size,), generator=rng)
    x = torch.stack([torch.from_numpy(data[i:i+length].astype(np.int64)) for i in starts.tolist()])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+length].astype(np.int64)) for i in starts.tolist()])
    return x.to(device), y.to(device), starts


def lr_at(index, peak, total, warmup):
    if index < warmup:
        return peak * index / warmup
    ratio = (index-warmup) / (total-warmup)
    return peak/10 + .5*(1+math.cos(math.pi*ratio))*(peak-peak/10)


def run(args):
    if args.profile == 'formal' and not args.device.startswith('cuda'):
        raise ValueError('Formal runs require a verified CUDA device; use --profile smoke for local checks.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no run started.')
    if args.dtype != 'float32' and not args.device.startswith('cuda'):
        raise ValueError('Mixed precision requires CUDA.')
    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bfloat16; choose float32 consistently for all runs.')
    smoke = args.profile == 'smoke'
    steps, interval, eval_iters = (3, 1, 2) if smoke else (5000, 250, 200)
    batch_size, block = (2, 16) if smoke else (64, 256)
    warmup = 1 if smoke else 100
    cfg = GPTConfig(variant=args.variant, vocab_size=65, n_layer=1 if smoke else 6,
                    n_head=6, n_embd=24 if smoke else 384, block_size=block,
                    dropout=.2, bias=False)
    torch.manual_seed(1337); np.random.seed(1337); random.seed(1337)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed_all(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if smoke:
        torch.set_num_threads(2)
    data_dir = ROOT/'data'/'shakespeare_char'
    data = {split: np.memmap(data_dir/f'{split}.bin', dtype=np.uint16, mode='r') for split in ('train','val')}
    for values in data.values():
        if len(values) <= block or int(values.max()) >= cfg.vocab_size:
            raise ValueError('Invalid dataset; run scripts/prepare_data.py first.')
    hashes = {f: hashlib.sha256((data_dir/f).read_bytes()).hexdigest() for f in ('train.bin','val.bin','input.txt')}
    model = GPT(cfg).to(args.device)
    optimizer = model.configure_optimizers(.1, args.lr, (.9,.99), 'cuda' if args.device.startswith('cuda') else 'cpu')
    context = lambda: (nullcontext() if args.dtype == 'float32' else torch.autocast('cuda', dtype=torch.bfloat16))
    # Sampling does not depend on parameter initialization, dropout, or evaluation calls.
    train_rng = torch.Generator().manual_seed(1338)
    torch.manual_seed(1339)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(1339)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(profile=args.profile, variant=args.variant, seed=1337,
                    data_seed=1338, dropout_seed=1339, optimizer_updates=steps,
                    batch_size=batch_size, eval_interval=interval, eval_iters=eval_iters,
                    learning_rate=args.lr, min_lr=args.lr/10, warmup_iters=warmup,
                    weight_decay=.1, betas=[.9,.99], grad_clip=1.,
                    model=asdict(cfg), dtype=args.dtype, device=args.device,
                    hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else platform.processor(),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    compile=False, tf32=True, data_sha256=hashes,
                    parameters=sum(p.numel() for p in model.parameters()),
                    upstream_commit='3adf61e154c3fe3fca428ad6bc3818b27a3b8291',
                    code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('model_variants.py','model.py','run_experiment.py')})
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    (out/'environment.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))+'\n')
    best, best_step = float('inf'), 0
    sample_digest = hashlib.sha256()
    start = time.monotonic()
    log = out/'metrics.jsonl'
    def emit(row):
        with log.open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
    @torch.no_grad()
    def evaluate():
        model.eval()
        result = {}
        for split, seed in (('train',2340), ('val',2341)):
            # Identical fixed evaluation batches for every checkpoint and variant.
            rng = torch.Generator().manual_seed(seed)
            losses=[]
            for _ in range(eval_iters):
                x,y,_=batch(data[split],batch_size,block,rng,args.device)
                with context(): _, loss=model(x,y)
                losses.append(float(loss))
            result[split+'_loss']=sum(losses)/len(losses)
        model.train()
        return result
    for completed in range(steps+1):
        if completed % interval == 0 or completed == steps:
            result=evaluate()
            emit(dict(kind='eval',step=completed,seconds=time.monotonic()-start,**result))
            print(f'{args.variant} step {completed}/{steps}: {result}',flush=True)
            if result['val_loss'] < best:
                best,best_step=result['val_loss'],completed
                torch.save(dict(model=model.state_dict(),model_config=asdict(cfg),step=completed),out/'best.pt')
        if completed == steps:
            break
        lr=lr_at(completed,args.lr,steps,warmup)
        for group in optimizer.param_groups: group['lr']=lr
        x,y,starts=batch(data['train'],batch_size,block,train_rng,args.device)
        sample_digest.update(starts.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        with context(): _,loss=model(x,y)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; retaining logs.')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        emit(dict(kind='update',step=completed+1,loss=float(loss.detach()),lr=lr,grad_norm=float(norm)))
    torch.save(dict(model=model.state_dict(),model_config=asdict(cfg),step=steps),out/'final.pt')
    summary=dict(status='complete',profile=args.profile,updates=steps,best_val_loss=best,
                 best_step=best_step,final_val_loss=result['val_loss'],
                 final_train_loss=result['train_loss'],train_sample_sha256=sample_digest.hexdigest(),
                 seconds=time.monotonic()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--profile',choices=('smoke','formal'),default='smoke')
    p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=('float32','bfloat16'),default='float32')
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if not math.isfinite(args.lr) or args.lr<=0: p.error('--lr must be positive and finite')
    run(args)

if __name__=='__main__': main()
