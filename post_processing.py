import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
# --- CONFIGURATION ---
MATRIX_FILE = "test_matrix.json"
RESULTS_DIR = "test_results"
ANALYSIS_DIR = "analysis"

# Assuming standard logging rate (e.g., 20 FPS = 0.05s per frame)
DELTA_T = 0.05 

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_kpis(telemetry_path):
    """ Parses a single run's telemetry and returns the scalar KPIs. """
    try:
        df = pd.read_csv(telemetry_path)
        
        # 1. Max Cross-Track Error
        max_cte = df['CTE'].abs().max() if 'CTE' in df.columns else 0.0
        
        # 2. Time to Line Crossing (Min_TLC)
        LANE_WIDTH_HALF = 1.75 
        lat_vel = df['CTE'].diff() / DELTA_T
        dist_to_boundary = LANE_WIDTH_HALF - df['CTE'].abs()
        lat_vel_safe = lat_vel.abs().replace(0, np.nan)
        # Clamping negative TLC to 0.0 to represent a physical lane departure
        min_tlc = (dist_to_boundary / lat_vel_safe).clip(lower=0.0).min()
        
        # 3. Deconstructed Jerk (Longitudinal and Lateral)
        jerk_long = df['Acc_X'].diff() / DELTA_T
        jerk_lat = df['Acc_Y'].diff() / DELTA_T
        
        # RMS Jerk
        rms_jerk_long = np.sqrt(np.mean(jerk_long.dropna()**2))
        rms_jerk_lat = np.sqrt(np.mean(jerk_lat.dropna()**2))
        
        # Max Absolute Jerk
        max_jerk_long = jerk_long.abs().max()
        max_jerk_lat = jerk_lat.abs().max()
        
        # 4. Lane Evasion Tracking (Using Native CARLA Sensor)
        is_invading = df['Lane_Invasion'].astype(bool) if 'Lane_Invasion' in df.columns else pd.Series([False]*len(df))
        evasion_starts = is_invading & (~is_invading.shift(1, fill_value=False))
        total_evasions = int(evasion_starts.sum())
        
        total_frames = len(df)
        ldr_percentage = (is_invading.sum() / total_frames) * 100.0 if total_frames > 0 else 0.0
        
        return {
            "Max_CTE": round(max_cte, 4),
            "Min_TLC": round(min_tlc, 4),
            "RMS_Jerk_Long": round(rms_jerk_long, 4),
            "RMS_Jerk_Lat": round(rms_jerk_lat, 4),
            "Max_Jerk_Long": round(max_jerk_long, 4),
            "Max_Jerk_Lat": round(max_jerk_lat, 4),
            "Total_Evasions": total_evasions,
            "LDR_Percent": round(ldr_percentage, 2)
        }
    except Exception as e:
        print(f"Error processing {telemetry_path}: {e}")
        return None


import statsmodels.formula.api as smf

def generate_interaction_regression(df):
    print("Running Multiple Regression with 2-Way Interactions...")
    
    outputs = [
        'Max_CTE', 'Min_TLC', 
        'RMS_Jerk_Long', 'RMS_Jerk_Lat', 
        'Max_Jerk_Long', 'Max_Jerk_Lat', 
        'Total_Evasions', 'LDR_Percent'
    ]
    
    # The formula automatically creates all main effects and all 2-way combinations
    formula_base = "(u_sun_alt + u_clouds + u_rain + u_wetness + u_fog_den)**2"
    
    summary_file = os.path.join(ANALYSIS_DIR, "thesis_regression_summaries.txt")
    
    with open(summary_file, 'w') as f:
        f.write("--- MULTIPLE REGRESSION INTERACTION ANALYSIS ---\n\n")
        
        for kpi in outputs:
            # We must drop NaNs for the specific KPI so the regression matrix aligns
            valid_df = df.dropna(subset=[kpi]).copy()
            
            # Variance check: Regression fails if the metric never changes (e.g. 0 evasions)
            if len(valid_df) > 1 and valid_df[kpi].std() > 0:
                formula = f"{kpi} ~ {formula_base}"
                model = smf.ols(formula=formula, data=valid_df).fit()
                
                # 1. Write the full academic summary to the text file
                f.write(f"===================================================\n")
                f.write(f"DEPENDENT VARIABLE: {kpi}\n")
                f.write(f"===================================================\n")
                f.write(model.summary().as_text() + "\n\n")
                
                # 2. Extract and print the dangerous interactions to the console
                print(f"\n[ {kpi} ] Significant Interactions (p < 0.05):")
                pvalues = model.pvalues
                
                # Filter for interaction terms (which contain a ':') and significant p-values
                interactions = pvalues[(pvalues.index.str.contains(':')) & (pvalues < 0.05)]
                
                if interactions.empty:
                    print("  -> None found.")
                else:
                    for term, p_val in interactions.items():
                        coef = model.params[term]
                        print(f"  -> {term}: Coef = {coef:.4f} (p = {p_val:.4f})")
            else:
                f.write(f"Skipping {kpi} due to zero variance in the data.\n\n")
                
    print(f"\nFull academic regression reports saved to: {summary_file}")

def build_dataset():
    print(f"Ingesting matrix and telemetry data...")
    with open(MATRIX_FILE, 'r') as f:
        test_matrix = json.load(f)
        
    dataset = []
    
    for run in test_matrix:
        run_name = run["run_name"]
        telemetry_file = os.path.join(RESULTS_DIR, run_name, "telemetry.csv")
        
        if os.path.exists(telemetry_file):
            kpis = calculate_kpis(telemetry_file)
            if kpis:
                # Extract the 5 independent mathematical ratios
                row = run["weather"]["regression_ratios"].copy()
                row["Run_Name"] = run_name
                row.update(kpis)
                dataset.append(row)
        else:
            print(f"Missing telemetry for: {run_name}")
            
    df = pd.DataFrame(dataset)
    df.set_index("Run_Name", inplace=True)
    return df

def generate_boxplots(df):
    print("Generating KPI Boxplots...")
    
    # Explicitly list the 8 target KPIs generated in calculate_kpis
    kpis = [
        'Max_CTE', 'Min_TLC', 
        'RMS_Jerk_Long', 'RMS_Jerk_Lat', 
        'Max_Jerk_Long', 'Max_Jerk_Lat', 
        'Total_Evasions', 'LDR_Percent'
    ]
    
    # Create a 2x4 grid for the 8 variables
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Distribution of Autonomous Performance KPIs across 70 Runs', fontsize=16)
    axes = axes.flatten()
    
    for i, kpi in enumerate(kpis):
        sns.boxplot(y=df[kpi], ax=axes[i], color='steelblue', width=0.4)
        sns.stripplot(y=df[kpi], ax=axes[i], color='darkred', alpha=0.5, size=5)
        axes[i].set_title(f'{kpi} Distribution')
        
        # Set dynamic y-axis labels based on the specific metric
        if kpi == 'Min_TLC':
            ylabel = 'Magnitude (Seconds)'
        elif kpi == 'Total_Evasions':
            ylabel = 'Event Count'
        elif kpi == 'LDR_Percent':
            ylabel = 'Percentage (%)'
        else:
            ylabel = 'Magnitude'
            
        axes[i].set_ylabel(ylabel)
        
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "kpi_boxplots.png"), dpi=300)
    plt.close()

def generate_marginal_analysis(df):
    print("Generating Marginal Correlation Matrices and Filtered Regressions...")
    
    inputs = ['u_sun_alt', 'u_clouds', 'u_rain', 'u_wetness', 'u_fog_den']
    outputs = [
        'Max_CTE', 'Min_TLC', 
        'RMS_Jerk_Long', 'RMS_Jerk_Lat', 
        'Max_Jerk_Long', 'Max_Jerk_Lat', 
        'Total_Evasions', 'LDR_Percent'
    ]
    
    # 1. Create DataFrames to store the raw r and p values
    r_matrix = pd.DataFrame(index=inputs, columns=outputs)
    p_matrix = pd.DataFrame(index=inputs, columns=outputs)
    annot_matrix = pd.DataFrame(index=inputs, columns=outputs)
    
    # Prepare the text file for the regression equations
    reg_summary_file = os.path.join(ANALYSIS_DIR, "marginal_regression_coefficients.txt")
    
    with open(reg_summary_file, 'w') as f:
        f.write("--- SIGNIFICANT MARGINAL REGRESSIONS (p < 0.05, |r| > 0.2) ---\n\n")
        
        for inp in inputs:
            for out in outputs:
                valid_data = df[[inp, out]].dropna() 
                
                # Variance check
                if len(valid_data) > 1 and valid_data[inp].std() > 0 and valid_data[out].std() > 0:
                    r, p = pearsonr(valid_data[inp], valid_data[out])
                else:
                    r, p = 0.0, 1.0 
                    
                # Populate the matrices
                r_matrix.loc[inp, out] = float(r)
                p_matrix.loc[inp, out] = float(p)
                annot_matrix.loc[inp, out] = f"{r:.2f}\n(p={p:.3f})"
                
                # 2. Filter for Significance and Impact
                if p < 0.05 and abs(r) > 0.2:
                    # Calculate the linear regression fit
                    slope, intercept, r_value, p_value, std_err = linregress(valid_data[inp], valid_data[out])
                    
                    # Log the exact equation to the text file
                    f.write(f"Pairing: {inp} -> {out}\n")
                    f.write(f"  Pearson r : {r:.4f} (p = {p:.4f})\n")
                    f.write(f"  Equation  : {out} = ({slope:.4f} * {inp}) + {intercept:.4f}\n")
                    f.write(f"  Std Error : {std_err:.4f}\n\n")
                    
                    # 3. Generate and save the scatter plot with regression line
                    plt.figure(figsize=(6, 5))
                    sns.regplot(
                        x=inp, y=out, data=valid_data, 
                        scatter_kws={'alpha': 0.6, 'color': 'steelblue'}, 
                        line_kws={'color': 'darkred'}
                    )
                    plt.title(f"Marginal Impact: {inp} on {out}\n$r={r:.2f}$, $p={p:.3f}$")
                    plt.xlabel(inp)
                    plt.ylabel(out)
                    plt.tight_layout()
                    
                    filename = f"marginal_{inp}_vs_{out}.png"
                    plt.savefig(os.path.join(ANALYSIS_DIR, filename), dpi=300)
                    plt.close()
                    
    # Ensure matrices are explicitly typed as floats for Seaborn
    r_matrix = r_matrix.astype(float)
    
    # 4. Save the raw numerical matrices to CSV
    r_matrix.to_csv(os.path.join(ANALYSIS_DIR, "correlation_r_matrix.csv"))
    p_matrix.to_csv(os.path.join(ANALYSIS_DIR, "correlation_p_matrix.csv"))
    
    # 5. Generate the main Heatmap (Keeping your original visual)
    plt.figure(figsize=(16, 6))
    sns.heatmap(r_matrix, annot=annot_matrix, fmt="", cmap='coolwarm', 
                vmin=-1, vmax=1, annot_kws={"size": 10}, cbar_kws={'label': 'Pearson Correlation (r)'})
    
    plt.title("Impact of Weather Parameters on AI Control KPIs")
    plt.xlabel("Key Performance Indicators (Outputs)")
    plt.ylabel("Independent Weather Ratios (Inputs)")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "input_output_correlation.png"), dpi=300)
    plt.close()
    
    print("Marginal analysis complete. Matrices, text summary, and plots saved.")


if __name__ == "__main__":
    ensure_dir(ANALYSIS_DIR)
    
    df = build_dataset()
    
    csv_path = os.path.join(ANALYSIS_DIR, "compiled_kpi_dataset.csv")
    df.to_csv(csv_path)
    print(f"\nSaved compiled dataset to {csv_path}")
    
    generate_boxplots(df)
    generate_marginal_analysis(df)
    generate_interaction_regression(df)
    print(f"\n✅ Analysis complete! Check the '{ANALYSIS_DIR}' folder for your statistical proofs.")