"""Run in an allocated GPU session before starting the formal comparison."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from model_variants import GPT, GPTConfig
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: open an allocated courses-gpu session first.')
# PyTorch 2.5 includes emulation in is_bf16_supported() by default.
# NVIDIA native BF16 requires compute capability 8.0 or later.
native_bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
dtype=torch.bfloat16 if native_bf16 else torch.float32
rows=[]
for variant in ('baseline','rmsnorm','swiglu','nope','rope','gqa'):
    torch.manual_seed(1337)
    torch.cuda.reset_peak_memory_stats()
    model=GPT(GPTConfig(variant=variant,n_layer=6,n_head=6,n_embd=384,
                        block_size=256,vocab_size=65,dropout=.2,bias=False)).cuda()
    opt=model.configure_optimizers(.1,.001,(.9,.99),'cuda')
    x=torch.randint(65,(64,256),device='cuda');y=torch.randint(65,(64,256),device='cuda')
    with torch.autocast('cuda',dtype=dtype,enabled=dtype==torch.bfloat16):
        _,loss=model(x,y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step();torch.cuda.synchronize()
    rows.append({'variant':variant,'loss':float(loss.detach()),
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    del model,opt,x,y,loss
    torch.cuda.empty_cache()
record={'purpose':'synthetic full-size GPU preflight, NOT formal results',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'recommended_dtype':str(dtype).split('.')[-1],'checks':rows}
out=ROOT/'results'/'gpu_preflight.json'
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
