import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import webbrowser

def generate_professional_3d_plots(csv_path="analysis/compiled_kpi_dataset.csv"):
    print(f"Loading actual run data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: Could not locate dataset at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    analysis_dir = "analysis"

    # 1. Create the environmental meshgrid (0.0 to 1.0)
    sun_range = np.linspace(0, 1, 50)
    wetness_range = np.linspace(0, 1, 50)
    X_sun, Y_wet = np.meshgrid(sun_range, wetness_range)

    # 2. Apply the strictly validated predictive equations
    Z_CTE = 1.1523 - 0.1885*(X_sun) + 0.9474*(Y_wet) - 0.8679*(X_sun * Y_wet)
    Z_TLC = 0.0294 - 0.0110*(X_sun) - 0.0266*(Y_wet) + 0.0282*(X_sun * Y_wet)

    # =========================================================================
    # PLOT 1: Maximum Cross-Track Error (CTE)
    # =========================================================================
    fig1 = go.Figure()

    # Add the continuous regression surface
    fig1.add_trace(go.Surface(
        z=Z_CTE, x=X_sun, y=Y_wet,
        colorscale='Reds',
        opacity=0.75,
        name='Regression Surface',
        showscale=True,
        colorbar=dict(title='Drift Severity', len=0.5, x=1.1)
    ))

    # Overlay the ACTUAL N=70 simulation runs as floating spheres
    fig1.add_trace(go.Scatter3d(
        x=df['u_sun_alt'], y=df['u_wetness'], z=df['Max_CTE'],
        mode='markers',
        marker=dict(size=4, color='black', line=dict(width=1, color='white')),
        name='Simulation Data',
        hovertemplate='Sun: %{x:.2f}<br>Wetness: %{y:.2f}<br>Max CTE: %{z:.3f}m<extra></extra>'
    ))

    fig1.update_layout(
        title='Interaction Effect: Environment on Spatial Drift (Max CTE)',
        scene=dict(
            xaxis_title='Sun Altitude (0=Night, 1=Day)',
            yaxis_title='Road Wetness (0=Dry, 1=Flooded)',
            zaxis_title='Predicted Max CTE (Meters)'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    cte_html_path = os.path.abspath(os.path.join(analysis_dir, "interactive_Max_CTE.html"))
    fig1.write_html(cte_html_path)
    print(f"Saved highly-interactive CTE plot to: {cte_html_path}")

    # =========================================================================
    # PLOT 2: Minimum Time to Line Crossing (TLC)
    # =========================================================================
    fig2 = go.Figure()

    fig2.add_trace(go.Surface(
        z=Z_TLC, x=X_sun, y=Y_wet,
        colorscale='RdBu', # Red for danger (low TLC), Blue for safe (high TLC)
        reversescale=True,
        opacity=0.75,
        name='Regression Surface',
        showscale=True,
        colorbar=dict(title='Buffer (s)', len=0.5, x=1.1)
    ))

    fig2.add_trace(go.Scatter3d(
        x=df['u_sun_alt'], y=df['u_wetness'], z=df['Min_TLC'],
        mode='markers',
        marker=dict(size=4, color='black', line=dict(width=1, color='white')),
        name='Simulation Data',
        hovertemplate='Sun: %{x:.2f}<br>Wetness: %{y:.2f}<br>Min TLC: %{z:.3f}s<extra></extra>'
    ))

    fig2.update_layout(
        title='Interaction Effect: Environment on Control Buffer (Min TLC)',
        scene=dict(
            xaxis_title='Sun Altitude (0=Night, 1=Day)',
            yaxis_title='Road Wetness (0=Dry, 1=Flooded)',
            zaxis_title='Predicted Min TLC (Seconds)'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    tlc_html_path = os.path.abspath(os.path.join(analysis_dir, "interactive_Min_TLC.html"))
    fig2.write_html(tlc_html_path)
    print(f"Saved highly-interactive TLC plot to: {tlc_html_path}")

    # Automatically open the results in your default web browser
    webbrowser.open('file://' + cte_html_path)
    webbrowser.open('file://' + tlc_html_path)

if __name__ == '__main__':
    generate_professional_3d_plots()