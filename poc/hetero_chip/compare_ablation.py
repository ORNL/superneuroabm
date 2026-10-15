"""Full device model vs static-only ablation, design by design (paired, same digits).

    python run_chip_poc.py && python run_chip_poc.py --static-only && python compare_ablation.py

Both runs use the same chip seed and agent ids, so the input spike trains and every
per-device constant are identical; the only difference is the in-run device physics.
The gap is what a fixed-weight simulator (e.g. NEST GPU) cannot represent.
"""

import json
import sys
from pathlib import Path

import numpy as np

from run_chip_poc import mcnemar_p

res = Path(__file__).resolve().parent / "results"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
full = json.loads((res / f"chip_seed{seed}_ranks1.json").read_text())
stat = json.loads((res / f"chip_seed{seed}_ranks1_static.json").read_text())
y = np.array(full["labels"])

print(f"{'design':<18}{'full':>8}{'static':>8}{'gap':>8}{'McNemar p':>12}")
for f, s in zip(full["designs"], stat["designs"]):
    ok_f = np.array(f["pred"]) == y
    ok_s = np.array(s["pred"]) == y
    print(f"{f['design']:<18}{ok_f.mean() * 100:>7.1f}%{ok_s.mean() * 100:>7.1f}%"
          f"{(ok_f.mean() - ok_s.mean()) * 100:>+7.1f}{mcnemar_p(ok_f, ok_s):>12.3g}")
