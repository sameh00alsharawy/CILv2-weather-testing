import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc

import numpy as np
from scipy.stats import pearsonr
# (Keep your other imports: json, os, pd, plt, sns, qmc)

def generate_statistics_figures(weather_data, output_dir="sampler_metrics"):
    """ Generates and saves distribution and correlation plots of the independent variables """
    print(f"\nGenerating statistical figures in '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # We plot the independent regression ratios to prove mathematical orthogonality.
    ratios_list = [w['regression_ratios'] for w in weather_data]
    df = pd.DataFrame(ratios_list)
    
    # --- 1. Plot Correlation Heatmap with p-values ---
    plt.figure(figsize=(10, 8))
    
    # Calculate standard correlation matrix
    correlation_matrix = df.corr()
    
    # Create an empty matrix to hold our custom "r \n (p=...)" strings
    annot_matrix = pd.DataFrame(np.empty_like(correlation_matrix, dtype=str), 
                                columns=df.columns, index=df.columns)
    
    # Loop through all pairs to calculate exact p-values
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                # Diagonal is perfectly correlated with itself
                annot_matrix.loc[col1, col2] = f"{correlation_matrix.loc[col1, col2]:.2f}\n(p=0.00)"
            else:
                # Calculate Pearson r and p-value
                r, p = pearsonr(df[col1], df[col2])
                annot_matrix.loc[col1, col2] = f"{r:.2f}\n(p={p:.2f})"
                
    # Pass the custom text matrix to the 'annot' parameter and set fmt="" so it reads the strings
    sns.heatmap(correlation_matrix, annot=annot_matrix, fmt="", cmap='coolwarm', 
                vmin=-1, vmax=1, annot_kws={"size": 9}) # Slightly reduced text size to fit both lines
    
    plt.title("Weather Parameter Correlation Matrix (r and p-values)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

    # --- 2. Plot Histograms for each independent parameter (1x5 Grid for 5 variables) ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Independent Ratio Distributions (Intact 5D Hypercube)', fontsize=16)
    axes = axes.flatten()
    
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], kde=True, ax=axes[i], color='seagreen', bins=10)
        axes[i].set_title(f"Distribution of {column}")
        axes[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(output_dir, "parameter_histograms.png"), dpi=300)
    plt.close()
    print("Figures saved successfully.")







def generate_lhs_weather(sample_size):
    print(f"Generating Intact 5D Latin Hypercube with {sample_size} samples...")
    
    # Generate perfect uniform samples between [0.0, 1.0) for 5 dimensions
    sampler = qmc.LatinHypercube(d=5)
    raw_samples = sampler.random(n=sample_size)
    
    valid_weathers = []
    
    for i, u in enumerate(raw_samples):
        u_sun_alt, u_clouds, u_rain, u_wetness, u_fog_den = u
        
        # 1. Base Variables
        sun_alt = -90.0 + (180.0 * u_sun_alt)
        clouds = 100.0 * u_clouds
        
        # 2. Constraint: Rain requires clouds
        rain = clouds * u_rain
        
        # 3. Constraint: Active heavy rain means the road must be wet
        if rain > 50.0:
            wetness = 30.0 + (70.0 * u_wetness)
        else:
            wetness = 100.0 * u_wetness
            
        # 4. Constraint: Prevent Unreal Engine shader white-out
        if rain > 80.0:
            fog_density = 80.0 * u_fog_den
        else:
            fog_density = 100.0 * u_fog_den
            
        weather_dict = {
            "name": f"LHS_{i:03d}",
            # --- Physical values injected into CARLA ---
            "sun_alt": round(sun_alt, 2),
            "sun_azimuth": 180.0,  # Hardcoded default
            "clouds": round(clouds, 2),
            "rain": round(rain, 2),
            "puddles": 0.0,        # Hardcoded default
            "wetness": round(wetness, 2),
            "fog_density": round(fog_density, 2),
            "fog_distance": 0.0,   # Hardcoded worst-case scenario
            # --- Independent ratios used for final Statistical Analysis ---
            "regression_ratios": {
                "u_sun_alt": round(u_sun_alt, 4),
                "u_clouds": round(u_clouds, 4),
                "u_rain": round(u_rain, 4),
                "u_wetness": round(u_wetness, 4),
                "u_fog_den": round(u_fog_den, 4)
            }
        }
        valid_weathers.append(weather_dict)
        
    print(f"Yield: {len(valid_weathers)} physically valid states (100% retention rate).")
    generate_statistics_figures(valid_weathers)
    return valid_weathers

def build_test_suite(sample_size, output_file="test_matrix.json"):
    town = "Town02"
    weathers = generate_lhs_weather(sample_size)

    test_matrix = []
    run_counter = 1

    print("Building execution matrix...")
    for weather in weathers:
        route_id = 0 
        run_name = f"run_{run_counter:03d}_{town}_rt{route_id}_{weather['name']}"
        
        run_config = {
            "run_name": run_name,
            "town": town,
            "route_id": route_id,
            "weather": weather
        }
        test_matrix.append(run_config)
        run_counter += 1

    with open(output_file, 'w') as f:
        json.dump(test_matrix, f, indent=4)
    
    print(f"\n✅ Success! Test suite with {len(test_matrix)} total runs saved to {output_file}")

if __name__ == "__main__":
    # 5D Matrix means 50 is your statistical floor. 60 gives you great density.
    build_test_suite(sample_size=70)