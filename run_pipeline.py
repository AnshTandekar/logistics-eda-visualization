"""Runs the full analysis. Usage: python run_pipeline.py
(Use --skip-simulation if data/shipments.csv already exists or you use your own data.)"""
import subprocess
import sys

STEPS = ["00_simulate_dataset", "01_eda_summary", "02_visualizations", "03_drivers_whatif"]
if "--skip-simulation" in sys.argv:
    STEPS = STEPS[1:]
for step in STEPS:
    print(f"\n===== {step} =====")
    subprocess.run([sys.executable, f"src/{step}.py"], check=True)
