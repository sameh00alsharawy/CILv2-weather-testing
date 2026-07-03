import carla
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import sys
import os

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ModuleNotFoundError:
    print("ERROR: 'agents' module not found. Make sure the CARLA agents folder is in this directory.")
    sys.exit(1)

def parse_route_xml(xml_path, route_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for route in root.findall('route'):
        if route.get('id') == str(route_id):
            waypoints = route.findall('route/waypoint') 
            if not waypoints:
                waypoints = route.findall('waypoint')
                
            start_wp = waypoints[0]
            end_wp = waypoints[1]
            
            start_transform = carla.Transform(
                carla.Location(x=float(start_wp.get('x')), y=float(start_wp.get('y')), z=float(start_wp.get('z')) + 0.5),
                carla.Rotation(pitch=float(start_wp.get('pitch')), yaw=float(start_wp.get('yaw')), roll=float(start_wp.get('roll')))
            )
            
            end_transform = carla.Transform(
                carla.Location(x=float(end_wp.get('x')), y=float(end_wp.get('y')), z=float(end_wp.get('z'))),
                carla.Rotation(pitch=float(end_wp.get('pitch')), yaw=float(end_wp.get('yaw')), roll=float(end_wp.get('roll')))
            )
            return start_transform, end_transform
    raise ValueError(f"Route ID {route_id} not found in {xml_path}")

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    xml_file = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\run_CARLA_driving\data\nocrash\Town02_navigation_lbc.xml"
    route_id_to_run = "18" 
    csv_file_path = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\debug_logs\telemetry.csv"

    # ==========================================
    # 2. LOAD GROUND TRUTH MAP
    # ==========================================
    print("Loading Map...")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    sim_world = client.load_world('Town02')
    
    start_transform, end_transform = parse_route_xml(xml_file, route_id_to_run)
    grp = GlobalRoutePlanner(sim_world.get_map(), 2.0)
    route_trace = grp.trace_route(start_transform.location, end_transform.location)
    
    map_x = [wp.transform.location.x for wp, _ in route_trace]
    map_y = [-wp.transform.location.y for wp, _ in route_trace]

    # ==========================================
    # 3. LOAD CSV TELEMETRY
    # ==========================================
    print("Loading CSV Data...")
    frames = []
    log_x, log_y = [], []
    log_colors = []
    
    # Tracking Logic Variables
    idx_current = []
    idx_scan_start = []
    idx_scan_end = []

    cmd_color_map = {'1': 'blue', '2': 'red', '3': 'green', '4': 'orange'}

    try:
        with open(csv_file_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append(int(row['Frame']))
                log_x.append(float(row['X']))
                log_y.append(-float(row['Y'])) 
                
                cmd = str(row['Command_Idx']).strip().replace('.0', '')
                log_colors.append(cmd_color_map.get(cmd, 'black'))
                
                # Load the tracking arrays
                idx_current.append(int(float(row['current_road_index'])))
                idx_scan_start.append(int(float(row['scan_start'])))
                idx_scan_end.append(int(float(row['scan_end'])))
                
    except FileNotFoundError:
        print(f"ERROR: Could not find {csv_file_path}.")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing column in CSV - {e}. Ensure your logger headers match exactly.")
        sys.exit(1)

    # ==========================================
    # 4. DRAW DUAL DASHBOARD
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # --- AXIS 1: SPATIAL MAP ---
    ax1.plot(map_x, map_y, c='lightgrey', linewidth=10, alpha=0.5, zorder=1)
    ax1.scatter(log_x, log_y, c=log_colors, s=20, edgecolor='black', linewidth=0.5, zorder=4)
    ax1.scatter(map_x[0], map_y[0], c='gold', marker='*', s=300, edgecolors='black', zorder=5)
    
    legend_patches = [
        mpatches.Patch(color='orange', label='Follow (4)'),
        mpatches.Patch(color='blue', label='Left (1)'),
        mpatches.Patch(color='red', label='Right (2)'),
        mpatches.Patch(color='green', label='Straight (3)')
    ]
    ax1.legend(handles=legend_patches, loc='best')
    ax1.set_title("Spatial Trajectory", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (Inverted, m)")
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- AXIS 2: LOGIC TRACKER ---
    ax2.plot(frames, idx_current, color='black', linewidth=2, label='Current Road Index')
    
    # Shade the lookahead window
    ax2.fill_between(frames, idx_scan_start, idx_scan_end, color='purple', alpha=0.2, label='Lookahead Scan Window')
    
    # Color-code the background of the graph based on the output command
    for i in range(1, len(frames)):
        ax2.axvspan(frames[i-1], frames[i], facecolor=log_colors[i], alpha=0.1)

    ax2.set_title("Navigation Tracking Logic Over Time", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Simulation Frame")
    ax2.set_ylabel("Waypoint Array Index")
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()