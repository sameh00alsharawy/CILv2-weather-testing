#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import collections
import datetime
import time
import math
import weakref
import json
import xml.etree.ElementTree as ET
import threading
import queue
import cv2
import argparse

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ModuleNotFoundError:
    print("ERROR: 'agents' module not found. Make sure the CARLA agents folder is in this directory.")
    sys.exit(1)

import carla
from carla import ColorConverter as cc

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
import csv 
import random

# ==============================================================================
# -- Asynchronous Image Saver --------------------------------------------------
# ==============================================================================
def worker_save_images(img_queue, base_dir):
    print(f"[Thread] Image writer started. Target directory: {base_dir}")
    while True:
        task = img_queue.get()
        if task is None:
            break
            
        frame_id, cam_name, array = task
        filename = os.path.join(base_dir, cam_name, f"{frame_id}.jpg")
        cv2.imwrite(filename, array)
        img_queue.task_done()
    print("[Thread] Image writer shutting down safely.")

# ==============================================================================
# -- AI Helpers ----------------------------------------------------------------
# ==============================================================================
def inject_weather(world, args):
    print("Injecting explicit weather parameters...")
    weather = carla.WeatherParameters(
        sun_altitude_angle=args.sun_alt,
        sun_azimuth_angle=args.sun_az,
        cloudiness=args.clouds,
        precipitation=args.rain,
        precipitation_deposits=args.puddles,
        wetness=args.wetness,
        fog_density=args.fog_den,
        fog_distance=args.fog_dist
    )
    world.set_weather(weather)
    return weather

def populate_city(client, world, num_vehicles=70, num_walkers=150):
    print(f"Spawning {num_vehicles} vehicles and {num_walkers} pedestrians...")
    
    blueprints = world.get_blueprint_library()
    vehicle_bps = blueprints.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    walker_bps = blueprints.filter('walker.pedestrian.*')
    
    spawned_vehicles = []
    spawned_walkers = []
    spawned_controllers = []

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        vehicle = world.try_spawn_actor(bp, spawn_points[i])
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            spawned_vehicles.append(vehicle)

    for _ in range(num_walkers):
        bp = random.choice(walker_bps)
        spawn_loc = world.get_random_location_from_navigation()
        if spawn_loc is None: continue
        spawn_transform = carla.Transform(spawn_loc)
        walker = world.try_spawn_actor(bp, spawn_transform)
        if walker is not None:
            spawned_walkers.append(walker)

    controller_bp = blueprints.find('controller.ai.walker')
    for walker in spawned_walkers:
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            spawned_controllers.append(controller)

    world.tick()

    for controller in spawned_controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.0 + random.random())

    return spawned_vehicles, spawned_walkers, spawned_controllers

def preprocess_image(raw_carla_image):
    array = np.frombuffer(raw_carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (raw_carla_image.height, raw_carla_image.width, 4))
    rgb_image = array[:, :, :3][:, :, ::-1] 
    
    image = Image.fromarray(rgb_image)
    image = image.resize((g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])).convert('RGB')
    
    tensor_image = TF.to_tensor(image)
    tensor_image = TF.normalize(tensor_image, 
                                mean=g_conf.IMG_NORMALIZATION['mean'], 
                                std=g_conf.IMG_NORMALIZATION['std'])
    return tensor_image

def parse_route_xml(xml_path, route_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for route in root.findall('route'):
        if route.get('id') == str(route_id):
            waypoints = route.findall('waypoint')
            start_wp = waypoints[0]
            end_wp = waypoints[1]
            
            start_transform = carla.Transform(
                carla.Location(x=float(start_wp.get('x')), y=float(start_wp.get('y')), z=float(start_wp.get('z')) ),
                carla.Rotation(pitch=float(start_wp.get('pitch')), yaw=float(start_wp.get('yaw')), roll=float(start_wp.get('roll')))
            )
            
            end_transform = carla.Transform(
                carla.Location(x=float(end_wp.get('x')), y=float(end_wp.get('y')), z=float(end_wp.get('z'))),
                carla.Rotation(pitch=float(end_wp.get('pitch')), yaw=float(end_wp.get('yaw')), roll=float(end_wp.get('roll')))
            )
            return start_transform, end_transform
    raise ValueError(f"Route ID {route_id} not found in {xml_path}")

# ==============================================================================
# -- World & HUD (Abbreviated for space, assume same as before) ----------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world):
        self.world = carla_world
        self.map = self.world.get_map()
        self.player = None

    def restart(self, spawn_transform=None):
        blueprint = self.world.get_blueprint_library().filter('vehicle.lincoln.mkz_2017')[0]
        blueprint.set_attribute('role_name', 'hero')
        if spawn_transform is None:
            spawn_points = self.map.get_spawn_points()
            spawn_transform = spawn_points[0] if spawn_points else carla.Transform()
        if self.player is not None:
            self.destroy()
        self.player = self.world.try_spawn_actor(blueprint, spawn_transform)

    def destroy(self):
        if self.player:
            self.player.destroy()

# ==============================================================================
# -- The Core Game Loop --------------------------------------------------------
# ==============================================================================

def game_loop(args):
    world = None
    ai_cameras = []
    
    termination_reason = "User Interrupt"
    run_start_time = time.time()
    total_distance_driven = 0.0
    last_location = None

    collision_history = []
    lane_invasions_history = []

    # 1. LOAD PYTORCH MODEL
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading AI on device: {device}")

    # Ensure your paths are correct for your local setup
    conf_file_path = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\_results\Ours\Town12346_5\config40.json"
    exp_dir = os.path.dirname(conf_file_path)

    with open(conf_file_path, 'r') as f:
        configuration_dict = json.loads(f.read())
        
    g_conf.immutable(False)
    merge_with_yaml(os.path.join(exp_dir, configuration_dict['yaml']), process_type='drive')
    os.environ["TRAINING_RESULTS_ROOT"] = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main"
    set_type_of_process('drive', root=os.environ["TRAINING_RESULTS_ROOT"])

    model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', f"{model.name}_{configuration_dict['checkpoint']}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # --- DIRECTORY SETUP ---
    debug_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(os.path.join(debug_dir, "center"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "right"), exist_ok=True)
    
    image_queue = queue.Queue()
    writer_thread = threading.Thread(target=worker_save_images, args=(image_queue, debug_dir))
    writer_thread.daemon = True
    writer_thread.start()

    try:
        # 2. CONNECT TO CARLA
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(60.0) 
        
        # Check if the map is already loaded to save 10+ seconds per run
        current_world = client.get_world()
        if args.town not in current_world.get_map().name:
            print(f"Loading {args.town}...")
            sim_world = client.load_world(args.town)
        else:
            print(f"{args.town} already active. Skipping map load...")
            sim_world = current_world

  

        # --- BULLETPROOF ZOMBIE CLEANUP ---
        print("Executing nuclear sweep for zombie actors...")
        actors = sim_world.get_actors()
        
        # 1. Explicitly hunt and destroy the ego vehicle and all other vehicles
        for actor in actors.filter('vehicle.*'):
            if actor.attributes.get('role_name') in ['ego_vehicle', 'hero']:
                print(f"Found zombie ego vehicle (ID: {actor.id}). Terminating...")
            actor.destroy() # Direct, synchronous kill command
                
        # 2. Sweep sensors and walkers
        for actor in actors.filter('sensor.*'): 
            actor.destroy()
        for actor in actors.filter('walker.*'): 
            actor.destroy()

        # 3. THE CRITICAL FIX: The Garbage Collection Pause
        print("Waiting for Unreal Engine to process physical deletions...")
        time.sleep(1.5) # Pauses Python to guarantee the server ticks and clears the spawn point
        # ---------------------------------
        
        # --- FREEZE TRAFFIC LIGHTS TO GREEN ---
        print("Freezing all traffic lights to GREEN to prevent timeouts...")
        for tl in actors.filter('traffic.traffic_light'):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        # ---------------------------------

        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)
        client.get_trafficmanager().set_synchronous_mode(True)

        inject_weather(sim_world, args)

        # 3. PARSE XML & GENERATE ROUTE
        xml_file = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\run_CARLA_driving\data\nocrash\Town02_navigation_lbc.xml"
        
        print(f"Parsing XML for Route ID: {args.route_id}...")
        start_transform, end_transform = parse_route_xml(xml_file, args.route_id)
        
        world = World(sim_world)
        world.restart(spawn_transform=start_transform)

        grp = GlobalRoutePlanner(sim_world.get_map(), 2.0)
        route_trace = grp.trace_route(start_transform.location, end_transform.location)
        current_route_index = 0

        # 4. SPAWN AI CAMERAS
        ai_bp = sim_world.get_blueprint_library().find('sensor.camera.rgb')
        ai_bp.set_attribute('image_size_x', '300')
        ai_bp.set_attribute('image_size_y', '300')
        ai_bp.set_attribute('fov', '60')

        cam_transforms = [
            carla.Transform(carla.Location(x=0.0, z=2.0), carla.Rotation(yaw=0.0)),   # 0: Center
            carla.Transform(carla.Location(x=0.0, z=2.0), carla.Rotation(yaw=-60.0)), # 1: Left
            carla.Transform(carla.Location(x=0.0, z=2.0), carla.Rotation(yaw=60.0))   # 2: Right
        ]

        ai_images = {0: None, 1: None, 2: None}

        for i, transform in enumerate(cam_transforms):
            cam = sim_world.spawn_actor(ai_bp, transform, attach_to=world.player)
            ai_cameras.append(cam)
            cam.listen(lambda img, idx=i: ai_images.__setitem__(idx, img))
        
        # --- LOGGER SETUP ---
        csv_file = open(os.path.join(debug_dir, "telemetry.csv"), mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'Frame', 'X', 'Y', 'Z', 'Ego_Yaw', 
            'Vel_X', 'Vel_Y', 'Vel_Z', 'Speed_kmh', 
            'Acc_X', 'Acc_Y', 'Acc_Z', 
            'CTE', 'Command_Idx', 
            'Raw_Steer', 'Raw_Throttle', 'Raw_Brake', 
            'Obstacles_JSON', 'Lane_Invasion'
        ])
        # --------------------

        bg_vehicles, bg_walkers, bg_controllers = populate_city(
            client=client, world=sim_world, num_vehicles=0, num_walkers=0
        )

        # --- COLLISION SENSOR SETUP ---
        def collision_callback(event):
            hit_type = event.other_actor.type_id
            if 'road' in hit_type or 'sidewalk' in hit_type or 'terrain' in hit_type: return
            collision_history.append(event)
            
        col_bp = sim_world.get_blueprint_library().find('sensor.other.collision')
        col_sensor = sim_world.spawn_actor(col_bp, carla.Transform(), attach_to=world.player)
        col_sensor.listen(lambda event: collision_callback(event))

        # --- LANE INVASION SETUP ---
        latest_lane_invasion = {"frame": -1}
        def lane_callback(event):
            lane_invasions_history.append(event)
            latest_lane_invasion["frame"] = event.frame
            
        lane_bp = sim_world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_sensor = sim_world.spawn_actor(lane_bp, carla.Transform(), attach_to=world.player)
        lane_sensor.listen(lambda event: lane_callback(event))

        # --- OBSTACLE SENSOR SETUP ---
        latest_obstacles = {}
        def obstacle_callback(event):
            target_actor = event.other_actor
            target_loc = target_actor.get_transform().location
            target_vel = target_actor.get_velocity()
            
            if event.frame not in latest_obstacles:
                latest_obstacles[event.frame] = []
                
            latest_obstacles[event.frame].append({
                'distance': round(event.distance, 2),
                'type': target_actor.type_id,
                'loc_x': round(target_loc.x, 3),
                'loc_y': round(target_loc.y, 3),
                'loc_z': round(target_loc.z, 3),
                'vel_x': round(target_vel.x, 3),
                'vel_y': round(target_vel.y, 3),
                'vel_z': round(target_vel.z, 3)
            })

        obs_bp = sim_world.get_blueprint_library().find('sensor.other.obstacle')
        obs_bp.set_attribute('distance', '20')
        obs_bp.set_attribute('hit_radius', '1.0')
        obs_bp.set_attribute('only_dynamics', 'True')
        obs_sensor = sim_world.spawn_actor(obs_bp, carla.Transform(), attach_to=world.player,)
        obs_sensor.listen(lambda event: obstacle_callback(event))

        # ---  THESE 4 LINES FIX THE FREEZE ---
        print("Warming up sensor buffers...")
        for _ in range(10):
            sim_world.tick()
        # -------------------------------------------

        # 5. AUTONOMOUS LOOP
        spectator = sim_world.get_spectator()
        print("Commencing automated run...")
        while True:
            frame_id = sim_world.tick()

            while any(img is None or img.frame < frame_id for img in ai_images.values()):
                pass

            car_transform = world.player.get_transform()
            car_location = car_transform.location
            ego_yaw = car_transform.rotation.yaw
            
            # ---  (The Chase Camera) ---
            spectator_transform = carla.Transform(
                car_transform.location - (car_transform.get_forward_vector() * 7.0) + carla.Location(z=3.0),
                car_transform.rotation
            )
            spectator.set_transform(spectator_transform)
            # --------------------------------------------

            if last_location is not None:
                total_distance_driven += car_location.distance(last_location)
            last_location = car_location

            # --- ASYNC IMAGE QUEUING ---
            cam_names = {0: 'center', 1: 'left', 2: 'right'}
            for idx, name in cam_names.items():
                raw_img = ai_images[idx]
                arr = np.frombuffer(raw_img.raw_data, dtype=np.dtype("uint8"))
                arr = np.reshape(arr, (raw_img.height, raw_img.width, 4))
                bgr_array = arr[:, :, :3]
                image_queue.put((frame_id, name, bgr_array))

            # --- DYNAMIC COMMAND GENERATION ---
            search_window = min(current_route_index + 4, len(route_trace))
            min_dist = float('inf')
            closest_idx = current_route_index
            
            for i in range(current_route_index, search_window):
                dist = car_location.distance(route_trace[i][0].transform.location)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
            current_route_index = closest_idx
            
            # END CONDITIONS
            dest_location = route_trace[-1][0].transform.location
            if car_location.distance(dest_location) < 5.0:
                print(f"\n🏁 Destination Reached Successfully! 🏁")
                termination_reason = "Success"
                break
            
            if len(collision_history) > 0:
                event = collision_history[0]
                print(f"\n💥 FATAL COLLISION DETECTED! 💥")
                termination_reason = f"Collision ({event.other_actor.type_id})"
                break

            cmd_idx = 4 
            scan_start = max(0, current_route_index - 2)
            scan_end = min(current_route_index + 6, len(route_trace))
            
            for idx in range(scan_start, scan_end):
                road_option = str(route_trace[idx][1])
                if "LEFT" in road_option:
                    cmd_idx = 1
                    break 
                elif "RIGHT" in road_option:
                    cmd_idx = 2
                    break
                elif "STRAIGHT" in road_option:
                    cmd_idx = 3
                    break

            if g_conf.DATA_COMMAND_ONE_HOT:
                one_hot = np.zeros(g_conf.DATA_COMMAND_CLASS_NUM, dtype=np.float32)
                if 0 <= (cmd_idx - 1) < g_conf.DATA_COMMAND_CLASS_NUM:
                    one_hot[cmd_idx - 1] = 1.0
                formatted_cmd = [torch.from_numpy(one_hot).unsqueeze(0).to(device)]
            else:
                formatted_cmd = [torch.tensor([[cmd_idx - 1]], dtype=torch.long).to(device)]

            processed_cams = [
                preprocess_image(ai_images[1]).unsqueeze(0).to(device),
                preprocess_image(ai_images[0]).unsqueeze(0).to(device),
                preprocess_image(ai_images[2]).unsqueeze(0).to(device)
            ]
            formatted_imgs = [processed_cams]

            # --- EXTRACT VELOCITY & ACCELERATION ---
            v = world.player.get_velocity()
            vel_x, vel_y, vel_z = v.x, v.y, v.z
            speed_ms = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
            
            acc = world.player.get_acceleration()
            acc_x, acc_y, acc_z = acc.x, acc.y, acc.z
            
            # --- CALCULATE TRUE CROSS-TRACK ERROR (CTE) ---
            wp = sim_world.get_map().get_waypoint(car_location)
            dx = car_location.x - wp.transform.location.x
            dy = car_location.y - wp.transform.location.y
            right_vector = wp.transform.get_right_vector()
            cte = (dx * right_vector.x) + (dy * right_vector.y)
            
            # --- AI INFERENCE ---
            min_speed = g_conf.DATA_NORMALIZATION['speed'][0]
            max_speed = g_conf.DATA_NORMALIZATION['speed'][1]
            normalized_speed = float(np.clip(abs(speed_ms - min_speed) / (max_speed - min_speed), 0.0, 1.0))
            formatted_speed = [torch.tensor([[normalized_speed]], dtype=torch.float32).to(device)]

            with torch.no_grad():
                driving_predictions, _, _ = model.forward_eval(
                    formatted_imgs, formatted_cmd, formatted_speed
                )
                outputs = driving_predictions.detach().cpu().numpy().squeeze()
            # --- APPLY DIRECT CONTROL ---
            control = carla.VehicleControl()
            if g_conf.ACCELERATION_AS_ACTION:
                raw_steer, acceleration = outputs[0], outputs[1]
                if acceleration >= 0.0:
                    raw_throttle, raw_brake = acceleration, 0.0
                else:
                    raw_brake, raw_throttle = np.abs(acceleration), 0.0
            else:
                raw_steer, raw_throttle, raw_brake = outputs[0], outputs[1], outputs[2]
                if raw_brake < 0.05: raw_brake = 0.0

            #raw_steer_clamped = 0.0 if abs(raw_steer) < 0.035 else raw_steer

            control.steer = float(np.clip(raw_steer, -1.0, 1.0))
            control.throttle = float(np.clip(raw_throttle, 0.0, 1.0))
            control.brake = float(np.clip(raw_brake, 0.0, 1.0))

            world.player.apply_control(control)

            # --- PREPARE SENSOR DATA FOR CSV ---
            obs_data = latest_obstacles.get(frame_id, [])
            obs_json_string = json.dumps(obs_data)
                
            keys_to_delete = [k for k in latest_obstacles.keys() if k < frame_id]
            for k in keys_to_delete: del latest_obstacles[k]

            is_lane_invasion = (latest_lane_invasion["frame"] == frame_id)

            # --- WRITE FULL TELEMETRY ROW ---
            csv_writer.writerow([
                frame_id, round(car_location.x, 3), round(car_location.y, 3), round(car_location.z, 3), round(ego_yaw, 3),
                round(vel_x, 3), round(vel_y, 3), round(vel_z, 3), round(speed_ms * 3.6, 2),
                round(acc_x, 3), round(acc_y, 3), round(acc_z, 3),
                round(cte, 4), cmd_idx,
                round(float(raw_steer), 4), round(float(raw_throttle), 4), round(float(raw_brake), 4),
                obs_json_string, is_lane_invasion
            ])


            print(f"Speed: {speed_ms*3.6:5.1f} km/h | Cmd: {cmd_idx} | Steer: {control.steer:5.2f} | Throt: {control.throttle:5.2f}  | Brake: {control.brake:5.2f}  ", end='\r')

    finally:
        print("\nCleaning up and generating summary...")
        
        summary = {
            "run_name": args.run_name,
            "termination_state": termination_reason,
            "run_time_seconds": round(time.time() - run_start_time, 2),
            "distance_driven_meters": round(total_distance_driven, 2),
            "infractions": {
                "collisions": len(collision_history),
                "lane_invasions": len(lane_invasions_history)
            }
        }
        with open(os.path.join(debug_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        image_queue.put(None)
        writer_thread.join()

        if 'csv_file' in locals(): csv_file.close()
        if 'bg_controllers' in locals():
            for controller in bg_controllers:
                controller.stop()
                controller.destroy()
        if 'bg_walkers' in locals():
            for walker in bg_walkers: walker.destroy()
        if 'bg_vehicles' in locals():
            client.apply_batch([carla.command.DestroyActor(x) for x in bg_vehicles])
            
        for cam in ai_cameras:
            if cam:
                cam.stop()
                cam.destroy()
                
        if 'col_sensor' in locals() and col_sensor is not None:
            col_sensor.stop()
            col_sensor.destroy()
            
        if 'lane_sensor' in locals() and lane_sensor is not None:
            lane_sensor.stop()
            lane_sensor.destroy()
            
        if 'obs_sensor' in locals() and obs_sensor is not None:
            obs_sensor.stop()
            obs_sensor.destroy()

        if world is not None:
            settings = sim_world.get_settings()
            settings.synchronous_mode = False
            sim_world.apply_settings(settings)
            world.destroy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CARLA AI Control Worker")
    
    # Run Identifiers
    parser.add_argument('--run_name', type=str, required=True, help="Unique name for this run")
    parser.add_argument('--output_dir', type=str, default="test_results", help="Base directory for outputs")
    
    # Environment Setup
    parser.add_argument('--town', type=str, required=True, help="CARLA Map to load (e.g., Town02)")
    parser.add_argument('--route_id', type=str, required=True, help="Route ID from the XML file")
    
    # Weather Parameters
    parser.add_argument('--sun_alt', type=float, required=True)
    parser.add_argument('--sun_az', type=float, required=True)
    parser.add_argument('--clouds', type=float, required=True)
    parser.add_argument('--rain', type=float, required=True)
    parser.add_argument('--puddles', type=float, required=True)
    parser.add_argument('--wetness', type=float, required=True)
    parser.add_argument('--fog_den', type=float, required=True)
    parser.add_argument('--fog_dist', type=float, required=True)
    
    args = parser.parse_args()
    game_loop(args)