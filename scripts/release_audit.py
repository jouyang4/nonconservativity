import subprocess, sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def run(args):
    return subprocess.call([sys.executable] + args, cwd=ROOT) == 0
audit = {
    "tests_passed": run(["-m", "pytest", "-q"]),
    "summary_stats_passed": run(["scripts/reproduce_summary_stats.py"]),
    "figure_scripts_passed": run(["scripts/make_figures.py"]),
    "examples_passed": run(["examples/compute_lambda_off_from_blocks.py"]) and run(["examples/conservative_field_control.py"]),
}
json.dump(audit, open(os.path.join(ROOT, "release_audit.json"), "w"), indent=1)
print(json.dumps(audit, indent=1))
sys.exit(0 if all(audit.values()) else 1)
