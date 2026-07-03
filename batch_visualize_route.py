import carla
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ModuleNotFoundError:
    print("ERROR: 'agents' module not found. Make sure the CARLA agents folder is in this directory.")
    sys.exit(1)

def get_all_routes_from_xml(xml_path):
    """ Parses the XML and returns a dictionary of all routes: {route_id: (start_transform, end_transform)} """
    routes_dict = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for route in root.findall('route'):
        route_id = route.get('id')
        
        # Handle different XML nesting structures
        waypoints = route.findall('route/waypoint') 
        if not waypoints:
            waypoints = route.findall('waypoint')
            
        if len(waypoints) >= 2:
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
            
            routes_dict[route_id] = (start_transform, end_transform)
            
    return routes_dict

def plot_and_save_route(route_id, route_trace, save_dir, town_name):
    """ Generates the plot and saves it directly to the hard drive """
    x_coords = []
    y_coords = []
    colors = []
    
    color_map = {
        "LEFT": "blue",
        "RIGHT": "red",
        "STRAIGHT": "green",
        "LANEFOLLOW": "gray"
    }

    for waypoint, road_option in route_trace:
        x_coords.append(waypoint.transform.location.x)
        y_coords.append(-waypoint.transform.location.y) # Invert Y for Matplotlib
        
        opt_str = str(road_option)
        c = "gray" 
        for key, val in color_map.items():
            if key in opt_str:
                c = val
                break
        colors.append(c)

    # Use a non-interactive backend to prevent RAM overflow when generating 100+ plots
    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(10, 8))
    
    plt.scatter(x_coords, y_coords, c=colors, s=15, zorder=2)
    plt.plot(x_coords, y_coords, c='black', alpha=0.3, zorder=1) 
    
    plt.scatter(x_coords[0], y_coords[0], c='gold', marker='*', s=200, edgecolors='black', label='Start', zorder=3)
    plt.scatter(x_coords[-1], y_coords[-1], c='purple', marker='X', s=150, edgecolors='black', label='End', zorder=3)
    
    legend_patches = [
        mpatches.Patch(color='gray', label='Lane Follow (Cmd: 4)'),
        mpatches.Patch(color='blue', label='Turn Left (Cmd: 1)'),
        mpatches.Patch(color='red', label='Turn Right (Cmd: 2)'),
        mpatches.Patch(color='green', label='Go Straight (Cmd: 3)')
    ]
    plt.legend(handles=legend_patches)
    
    plt.title(f"{town_name} - Route {route_id}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate (Inverted)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') 
    
    # Save and aggressively close the figure to free up memory
    save_path = os.path.join(save_dir, f"{town_name}_Route_{route_id.zfill(2)}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    base_data_dir = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\run_CARLA_driving\data\nocrash"
    output_dir = os.path.join(os.getcwd(), "route_visualizations")
    
    # Define the towns and their specific XML files
    town_configs = {
        "Town01": os.path.join(base_data_dir, "Town01_navigation_lbc.xml"),
        "Town02": os.path.join(base_data_dir, "Town02_navigation_lbc.xml")
    }

    # Connect to CARLA once
    print("Connecting to CARLA Server...")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)

    # ==========================================
    # 2. BATCH PROCESS TOWNS
    # ==========================================
    for town_name, xml_path in town_configs.items():
        if not os.path.exists(xml_path):
            print(f"Skipping {town_name}: XML file not found at {xml_path}")
            continue

        print(f"\nLoading {town_name}...")
        sim_world = client.load_world(town_name)
        grp = GlobalRoutePlanner(sim_world.get_map(), 2.0)
        
        # Create output folder for this town
        town_out_dir = os.path.join(output_dir, town_name)
        os.makedirs(town_out_dir, exist_ok=True)
        
        # Parse all routes for this town
        routes = get_all_routes_from_xml(xml_path)
        print(f"Found {len(routes)} routes in {town_name}. Generating plots...")
        
        for route_id, (start_tf, end_tf) in routes.items():
            print(f"  -> Plotting Route {route_id}...", end='\r')
            try:
                route_trace = grp.trace_route(start_tf.location, end_tf.location)
                plot_and_save_route(route_id, route_trace, town_out_dir, town_name)
            except Exception as e:
                print(f"\n  [!] Failed to plot Route {route_id}: {e}")
                
        print(f"\nFinished {town_name}! Images saved to: {town_out_dir}")

    print("\nBatch generation complete.")

if __name__ == '__main__':
    main()