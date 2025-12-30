
import subprocess
import time
import json
import os

print("=== STARTING SIMULATED TRADING SESSION ===")

# Clean start
if os.path.exists("data/paper_trade_state.json"):
    os.remove("data/paper_trade_state.json")

for i in range(1, 6):
    print(f"\n--- Cycle {i}/5 ---")
    
    # Run the bot for one cycle
    with open("sim_log.txt", "a") as f:
        f.write(f"\n--- Cycle {i}/5 ---\n")
        subprocess.run(
            ["./venv/bin/python", "main_v2.py", "--paper-trade", "--once"], 
            stdout=f,
            stderr=subprocess.STDOUT
        )
            
    print(f"Cycle {i} complete")
            
    # Simulate time gap (shortened for demo)
    time.sleep(2)

print("\n=== SESSION COMPLETE ===")
