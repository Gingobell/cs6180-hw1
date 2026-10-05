"""Tiny synthetic CPU check; not a formal assignment experiment."""
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from model import GPT, GPTConfig

torch.manual_seed(1337)
torch.set_num_threads(2)
model = GPT(GPTConfig(block_size=16, vocab_size=32, n_layer=1,
                      n_head=2, n_embd=32, dropout=0, bias=False))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(32, (2, 16))
y = torch.randint(32, (2, 16))
losses = []
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)
    optimizer.step()
    losses.append(loss.item())

result = dict(status="passed", purpose="synthetic smoke check only",
              formal_experiment=False, device="cpu", updates=3,
              torch_version=torch.__version__, machine=platform.machine(),
              mps_available=torch.backends.mps.is_available(), losses=losses)
out = ROOT / "results/smoke/baseline.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
