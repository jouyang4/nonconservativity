import subprocess, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def run(args): print("\n$ " + " ".join(args)); subprocess.check_call(args, cwd=ROOT)
run([sys.executable, "-m", "pytest", "-q"])
run([sys.executable, "scripts/reproduce_summary_stats.py"])
run([sys.executable, "scripts/make_figures.py"])
run([sys.executable, "examples/compute_lambda_off_from_blocks.py"])
run([sys.executable, "examples/conservative_field_control.py"])
print("PASS")
