import carla
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys
import os

# Ensure the agents module can be found
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
    # 1. Configuration
    xml_file = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\run_CARLA_driving\data\nocrash\Town02_navigation_lbc.xml"
    route_id_to_run = "2" # Change this to visualize different routes
    
    # 2. Connect to CARLA and load map
    print(f"Connecting to CARLA to map Route {route_id_to_run}...")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    sim_world = client.load_world('Town02')
    
    # 3. Parse XML and Trace Route
    start_transform, end_transform = parse_route_xml(xml_file, route_id_to_run)
    grp = GlobalRoutePlanner(sim_world.get_map(), 2.0)
    route_trace = grp.trace_route(start_transform.location, end_transform.location)
    
    # 4. Extract data for plotting
    x_coords = []
    y_coords = []
    colors = []
    
    # Mapping CARLA RoadOptions to colors
    color_map = {
        "LEFT": "blue",
        "RIGHT": "red",
        "STRAIGHT": "green",
        "LANEFOLLOW": "gray",
        "CHANGELANELEFT": "cyan",
        "CHANGELANERIGHT": "magenta"
    }

    for waypoint, road_option in route_trace:
        x_coords.append(waypoint.transform.location.x)
        # Invert Y axis because CARLA's Y axis and Matplotlib's Y axis are mirrored
        y_coords.append(-waypoint.transform.location.y) 
        
        # Determine color based on the RoadOption string
        opt_str = str(road_option)
        c = "gray" # Default
        for key, val in color_map.items():
            if key in opt_str:
                c = val
                break
        colors.append(c)

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, c=colors, s=15, zorder=2)
    plt.plot(x_coords, y_coords, c='black', alpha=0.3, zorder=1) # Draw the path line
    
    # Mark Start and End
    plt.scatter(x_coords[0], y_coords[0], c='gold', marker='*', s=200, edgecolors='black', label='Start', zorder=3)
    plt.scatter(x_coords[-1], y_coords[-1], c='purple', marker='X', s=150, edgecolors='black', label='End', zorder=3)
    
    # Create custom legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='gray', label='Lane Follow (Cmd: 4)'),
        mpatches.Patch(color='blue', label='Turn Left (Cmd: 1)'),
        mpatches.Patch(color='red', label='Turn Right (Cmd: 2)'),
        mpatches.Patch(color='green', label='Go Straight (Cmd: 3)')
    ]
    plt.legend(handles=legend_patches)
    
    plt.title(f"CARLA Global Route Planner - Route {route_id_to_run}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate (Inverted)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # Keep aspect ratio square so turns look accurate
    plt.show()

if __name__ == '__main__':
    main()
