import pandas as pd
import statsmodels.formula.api as smf
import argparse
import sys
import os

def run_predictive_models(csv_path):
    print(f"Loading telemetry data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: Could not locate data at {csv_path}. Please check the path.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Ensure column names match exactly. Adjust these strings if your dataframe uses different casing.
    print("\n==============================================================================")
    print("GROUP A: SPATIAL TRACKING (Strict Model: Sun & Wetness Only)")
    print("==============================================================================\n")
    
    group_a_kpis = ['Max_CTE', 'Min_TLC']
    formula_a = "{} ~ u_sun_alt * u_wetness" 

    for kpi in group_a_kpis:
        if kpi in df.columns:
            print(f"--- MODEL FOR: {kpi} ---")
            model = smf.ols(formula=formula_a.format(kpi), data=df).fit()
            print(model.summary())
            print("\n" + "="*78 + "\n")
        else:
            print(f"[Warning] Column '{kpi}' not found in dataframe.")

    print("\n==============================================================================")
    print("GROUP B: BOUNDARY FAILURES (Nuanced Model: Sun, Wetness, & Clouds)")
    print("==============================================================================\n")

    group_b_kpis = ['Total_Evasions', 'LDR_Percent']
    formula_b = "{} ~ u_sun_alt + u_wetness + u_clouds + u_sun_alt:u_wetness + u_clouds:u_wetness + u_sun_alt:u_clouds"

    for kpi in group_b_kpis:
        if kpi in df.columns:
            print(f"--- MODEL FOR: {kpi} ---")
            model = smf.ols(formula=formula_b.format(kpi), data=df).fit()
            print(model.summary())
            print("\n" + "="*78 + "\n")
        else:
            print(f"[Warning] Column '{kpi}' not found in dataframe.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute OLS Multiple Regression for CILv2 KPIs.")
    # You can change the default below to wherever your N=70 dataset actually lives
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the aggregated master CSV file")
    
    args = parser.parse_args()
    run_predictive_models(args.csv_path)