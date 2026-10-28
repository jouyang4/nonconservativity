import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nonconservativity.metrics import lambda_off, IID_BASELINE

d = np.load(os.path.join(os.path.dirname(__file__), "example_blocks.npz"))
J_ij, J_ji = d["J_ij"], d["J_ji"]
print("pairs=%d" % len(J_ij))
val = lambda_off(J_ij, J_ji)
print("lambda_off=%.4f iid_baseline=%.3f" % (val, IID_BASELINE))
