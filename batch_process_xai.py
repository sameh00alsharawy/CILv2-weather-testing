import os
import sys
import subprocess
import pandas as pd
import time

def run_xai_batch():
    # Define file paths
    csv_path = os.path.join("analysis", "outlier_runs_table.csv")
    xai_script_path = "extract_batch_gradcam.py"

    print("--- XAI Batch Processor ---")
    
    # Pre-flight checks
    if not os.path.exists(csv_path):
        print(f"Error: Could not locate {csv_path}. Make sure the file exists.")
        return
    if not os.path.exists(xai_script_path):
        print(f"Error: Could not locate {xai_script_path} in the root directory.")
        return

    # Ingest the outlier table
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    if 'Run_Name' not in df.columns:
        print("Error: 'Run_Name' column is missing from the CSV.")
        return

    # Extract unique runs to avoid redundant processing
    target_runs = df['Run_Name'].unique()
    total_runs = len(target_runs)
    
    print(f"Loaded {total_runs} unique outlier runs for processing.\n")

    # Execute the batch sequence
    for index, run_name in enumerate(target_runs, 1):
        print("=" * 60)
        print(f"[{index}/{total_runs}] Initiating XAI Diagnostic for: {run_name}")
        
        # Build the command using the active Python executable
        cmd = [
            sys.executable, 
            xai_script_path, 
            "--run_name", 
            str(run_name)
        ]

        start_time = time.time()
        try:
            # Execute the script. stdout and stderr will print naturally to the console.
            subprocess.run(cmd, check=True)
            
            elapsed_time = time.time() - start_time
            print(f"✅ Successfully completed {run_name} in {elapsed_time:.1f} seconds.")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {xai_script_path} crashed while processing {run_name}.")
            print(f"Exit Code: {e.returncode}")
            print("Continuing to the next run in the queue...\n")
            
        except KeyboardInterrupt:
            print("\n⚠️ Batch process manually interrupted by user. Exiting safely.")
            sys.exit(0)

        # Brief cooldown between heavy GPU loads
        time.sleep(1.0)

    print("=" * 60)
    print("🏁 XAI Master Panel generation sequence complete!")

if __name__ == "__main__":
    run_xai_batch()