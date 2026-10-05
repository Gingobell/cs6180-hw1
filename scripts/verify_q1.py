"""Verify Q1 with asymmetric kernels and arbitrary upstream gradients on CPU."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.manual_seed(1337)
max_errors = dict(forward=0.0, autograd=0.0, transpose=0.0,
                  flipped_correlation=0.0, finite_difference=0.0)
for trial in range(5):
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    g = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    patches = torch.stack([x[0, 0, i:i+3, j:j+3].reshape(9)
                           for i in range(2) for j in range(2)], dim=1)
    y = F.conv2d(x, k)
    matrix_y = (k.reshape(1, 9) @ patches).reshape_as(y)
    dx = torch.zeros_like(x)
    for i in range(2):
        for j in range(2):
            dx[0, 0, i:i+3, j:j+3] += g[0, 0, i, j] * k[0, 0]
    (y * g).sum().backward()
    finite = torch.zeros_like(x)
    eps = 1e-6
    for u in range(4):
        for v in range(4):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus[0, 0, u, v] += eps
            minus[0, 0, u, v] -= eps
            finite[0, 0, u, v] = ((F.conv2d(plus, k) * g).sum()
                                  - (F.conv2d(minus, k) * g).sum()) / (2 * eps)
    pairs = dict(forward=(matrix_y, y), autograd=(dx, x.grad),
                 transpose=(dx, F.conv_transpose2d(g, k)),
                 flipped_correlation=(dx, F.conv2d(g, k.flip(-2, -1), padding=2)),
                 finite_difference=(dx, finite))
    for name, (a, b) in pairs.items():
        error = (a - b).abs().max().item()
        max_errors[name] = max(max_errors[name], error)
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)

result = dict(status="passed", trials=5, dtype="float64", seed=1337,
              torch_version=torch.__version__, max_absolute_errors=max_errors)
output = Path(__file__).resolve().parents[1] / "results/q1_verification.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
