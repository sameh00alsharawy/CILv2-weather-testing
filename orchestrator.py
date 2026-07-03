import json
import os
import subprocess
import sys
import time

# --- CONFIGURATION ---
MATRIX_FILE = "test_matrix.json"
LOG_FILE = "progress_log.json"
WORKER_SCRIPT = "unified_ai_control.py"
MAX_RUNTIME_SECONDS = 300  # 5 minutes max per run to prevent infinite hanging

def load_progress():
    """Loads the progress log to resume interrupted batches."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress_dict):
    """Saves the current state to disk."""
    with open(LOG_FILE, 'w') as f:
        json.dump(progress_dict, f, indent=4)

def run_orchestrator():
    if not os.path.exists(MATRIX_FILE):
        print(f"Error: {MATRIX_FILE} not found. Run sampler.py first.")
        return

    with open(MATRIX_FILE, 'r') as f:
        test_matrix = json.load(f)

    progress = load_progress()
    total_runs = len(test_matrix)
    completed_runs = len([v for v in progress.values() if v in ["SUCCESS", "FAILED_TIMEOUT"]])

    print(f"--- CARLA Validation Orchestrator ---")
    print(f"Loaded {total_runs} total runs.")
    print(f"Resuming progress: {completed_runs}/{total_runs} already completed.\n")

    for index, run_config in enumerate(test_matrix):
        run_name = run_config["run_name"]
        
        # 1. Skip already completed runs
        if progress.get(run_name) in ["SUCCESS", "FAILED_TIMEOUT"]:
            continue

        print(f"[{index + 1}/{total_runs}] Starting: {run_name}")
        
        # 2. Build the command-line arguments
        w = run_config["weather"]
        cmd = [
            sys.executable, WORKER_SCRIPT,
            "--run_name", str(run_name),
            "--town", str(run_config["town"]),
            "--route_id", str(run_config["route_id"]),
            "--sun_alt", str(w["sun_alt"]),
            "--sun_az", str(w["sun_azimuth"]),
            "--clouds", str(w["clouds"]),
            "--rain", str(w["rain"]),
            "--puddles", str(w["puddles"]),
            "--wetness", str(w["wetness"]),
            "--fog_den", str(w["fog_density"]),
            "--fog_dist", str(w["fog_distance"])
        ]

        # 3. Execute the worker process
        start_time = time.time()
        try:
            # We redirect stdout/stderr to completely isolate the console output,
            # or you can leave them as None to see the worker's prints.
            process = subprocess.run(
                cmd, 
                timeout=MAX_RUNTIME_SECONDS,
                check=True 
            )
            
            progress[run_name] = "SUCCESS"
            print(f"   -> Completed in {round(time.time() - start_time, 1)}s")

        except subprocess.TimeoutExpired:
            print(f"   -> ⚠️ TIMEOUT: Run exceeded {MAX_RUNTIME_SECONDS}s. CARLA likely froze.")
            print(f"   -> Subprocess aggressively killed. Moving to next run.")
            progress[run_name] = "FAILED_TIMEOUT"

        except subprocess.CalledProcessError as e:
            # The script crashed (e.g., failed to connect to CARLA, missing CUDA)
            print(f"\n❌ FATAL WORKER CRASH (Exit Code {e.returncode})")
            print(f"   -> This usually means the CARLA Server is down or unreachable.")
            print(f"   -> Saving progress and halting Orchestrator. Please investigate.\n")
            progress[run_name] = "CRASHED"
            save_progress(progress)
            sys.exit(1)

        # Save progress after every single run
        save_progress(progress)
        
        # --- THE NETWORK COOLDOWN FIX ---
        print("   -> Cooling down network sockets for 4 seconds...")
        time.sleep(4.0)
        print("-" * 50 + "\n")

    print("\n🏁 All runs in the test matrix have been processed!")

if __name__ == "__main__":
    run_orchestrator()