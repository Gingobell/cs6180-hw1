"""Independent numerical/structural checks, not formal training evidence."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from model import GPT as OriginalGPT, GPTConfig as OriginalConfig
from model_variants import GPT, GPTConfig, RMSNorm, apply_rope
from run_experiment import run, VARIANTS

torch.set_num_threads(2)
torch.manual_seed(77)
small=dict(n_layer=2,n_head=6,n_embd=24,block_size=16,vocab_size=65,dropout=0.,bias=False)
x=torch.randint(65,(2,12)); y=torch.randint(65,(2,12))
a=OriginalGPT(OriginalConfig(**small)).double()
b=GPT(GPTConfig(**small)).double();b.load_state_dict(a.state_dict())
la,loss_a=a(x,y);lb,loss_b=b(x,y)
torch.testing.assert_close(la,lb,rtol=0,atol=0)
loss_a.backward();loss_b.backward()
for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
record={'baseline_matches_upstream_logits_and_gradients':'exact','variants':{}}
# Independent scalar RMS normalization and gradient finite difference.
z=torch.randn(2,7,dtype=torch.float64,requires_grad=True)
norm=RMSNorm(7).double()
expected=z/(z.square().sum(-1,keepdim=True)/7+1e-5).sqrt()
torch.testing.assert_close(norm(z),expected)
assert torch.autograd.gradcheck(norm,(z,))
# Rotation norm and shared-position-shift dot-product invariance.
q=torch.randn(2,3,8,4,dtype=torch.float64);k=torch.randn_like(q)
torch.testing.assert_close(apply_rope(q).square().sum(-1),q.square().sum(-1))
torch.testing.assert_close(apply_rope(q)@apply_rope(k).transpose(-2,-1),
                           apply_rope(q,13)@apply_rope(k,13).transpose(-2,-1))
for variant in VARIANTS:
    m=GPT(GPTConfig(variant=variant,**small)).double().eval()
    logits,loss=m(x,y)
    changed=x.clone(); changed[:,6:]=(changed[:,6:]+7)%65
    future_logits,_=m(changed,y)
    torch.testing.assert_close(logits[:,:6],future_logits[:,:6],rtol=0,atol=0)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if variant in ('nope','rope'): assert 'transformer.wpe.weight' not in m.state_dict()
    if variant=='swiglu':
        base_mlp=a.transformer.h[0].mlp
        assert sum(p.numel() for p in m.transformer.h[0].mlp.parameters())==sum(p.numel() for p in base_mlp.parameters())
    if variant=='gqa':
        att=m.transformer.h[0].attn
        inp=torch.randn(2,9,24,dtype=torch.float64)
        qq,kk,vv=att.c_attn(inp).split([24,12,12],dim=-1)
        qq=qq.view(2,9,6,4).transpose(1,2)
        kk=kk.view(2,9,3,4).transpose(1,2)
        vv=vv.view(2,9,3,4).transpose(1,2)
        heads=[]
        mask=torch.ones(9,9,dtype=torch.bool).tril()
        for h in range(6):
            scores=(qq[:,h]@kk[:,h//2].transpose(-2,-1))/2
            weights=scores.masked_fill(~mask,float('-inf')).softmax(-1)
            heads.append(weights@vv[:,h//2])
        manual=att.c_proj(torch.stack(heads,dim=1).transpose(1,2).reshape(2,9,24))
        torch.testing.assert_close(att(inp),manual,rtol=1e-10,atol=1e-12)
    # Count full-size parameters without allocating large tensors.
    with torch.device('meta'):
        full=GPT(GPTConfig(variant=variant,n_layer=6,n_head=6,n_embd=384,
                           block_size=256,vocab_size=65,dropout=.2,bias=False))
    record['variants'][variant]={'causal_and_finite_gradients':'passed','parameters':sum(p.numel() for p in full.parameters())}
record['rmsnorm_formula_and_gradcheck']='passed'
record['rope_norm_and_relative_position_invariance']='passed'
record['gqa_per_head_reference']='passed'
record['swiglu_mlp_parameter_match']='exact'
# Actual tiny Shakespeare runs: verifies pipeline, exact updates and paired sampling.
with tempfile.TemporaryDirectory(prefix='hw1-q4-check-') as d:
    summaries={}
    for variant in VARIANTS:
        folder=Path(d)/variant
        summaries[variant]=run(SimpleNamespace(variant=variant,profile='smoke',device='cpu',dtype='float32',lr=.001,out=str(folder)))
        rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows if r['kind']=='update']==[1,2,3]
        assert [r['step'] for r in rows if r['kind']=='eval']==[0,1,2,3]
        saved=torch.load(folder/'final.pt',weights_only=True)
        assert saved['step']==3
        # Retain small-run logs/config/summary, not temporary checkpoints.
        dest=ROOT/'results'/'smoke'/'q4'/variant
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','metrics.jsonl','summary.json','environment.txt'):
            (dest/name).write_bytes((folder/name).read_bytes())
    assert len({s['train_sample_sha256'] for s in summaries.values()})==1
    repeat=run(SimpleNamespace(variant='baseline',profile='smoke',device='cpu',dtype='float32',lr=.001,out=str(Path(d)/'repeat')))
    for key in ('best_val_loss','final_val_loss','final_train_loss','train_sample_sha256'):
        assert repeat[key]==summaries['baseline'][key]
record['pipeline']='six real-data smoke runs, exactly 3 updates each; equal data batches; baseline repeat exact'
(ROOT/'results'/'q4_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
