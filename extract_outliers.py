import pandas as pd
import numpy as np

def extract_outliers():
    print("Loading compiled KPI dataset...")
    # Load the dataset we generated previously
    df = pd.read_csv("analysis/compiled_kpi_dataset.csv", index_col="Run_Name")
    
    # The 8 KPIs to check
    kpis = [
        'Max_CTE', 'Min_TLC', 
        'RMS_Jerk_Long', 'RMS_Jerk_Lat', 
        'Max_Jerk_Long', 'Max_Jerk_Lat', 
        'Total_Evasions', 'LDR_Percent'
    ]
    
    # Create an empty boolean mask of the same length as the dataframe, starting as False
    is_outlier_anywhere = pd.Series(False, index=df.index)
    
    # Dictionary to keep track of *why* it's an outlier (optional, but helpful)
    outlier_reasons = {run: [] for run in df.index}

    for kpi in kpis:
        # Calculate Q1, Q3, and IQR
        Q1 = df[kpi].quantile(0.25)
        Q3 = df[kpi].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find which rows are outside the bounds for this specific KPI
        outlier_mask = (df[kpi] < lower_bound) | (df[kpi] > upper_bound)
        
        # Record the reason
        for run in df[outlier_mask].index:
            outlier_reasons[run].append(kpi)
            
        # Update the master mask (True if it was an outlier previously OR is an outlier now)
        is_outlier_anywhere = is_outlier_anywhere | outlier_mask

    # Filter the dataframe to only include the outlier runs
    outliers_df = df[is_outlier_anywhere].copy()
    
    # Add a column detailing which KPIs triggered the outlier status
    outliers_df['Outlier_Triggered_By'] = [", ".join(outlier_reasons[run]) for run in outliers_df.index]
    
    # Save the table
    output_path = "analysis/outlier_runs_table.csv"
    
    # Reorder columns so the KPIs and the trigger reason are front and center
    cols = kpis + ['Outlier_Triggered_By'] + [c for c in outliers_df.columns if c not in kpis and c != 'Outlier_Triggered_By']
    outliers_df = outliers_df[cols]
    
    outliers_df.to_csv(output_path)
    
    print(f"\nFound {len(outliers_df)} total outlier runs.")
    print(f"Table saved successfully to: {output_path}")
    
    # Display a quick preview in the console
    print("\nPreview of extracted outliers:")
    print(outliers_df[['Outlier_Triggered_By']].head())

if __name__ == "__main__":
    extract_outliers()