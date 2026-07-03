import os
import sys
import json
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import argparse

# Import configurations
from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from _utils.grad_cam.grad_cam import GradCAM

# ==============================================================================
# -- Hook Vault ----------------------------------------------------------------
# ==============================================================================
class AttentionForensics:
    """An isolated vault to safely catch and guard tensors."""
    def __init__(self):
        self.attn_weights = None
        self.output_tensor = None

    def hook(self, module, input, output):
        self.attn_weights = output[1]
        if output[0].requires_grad:
            self.output_tensor = output[0]
            self.output_tensor.retain_grad()
            
    def reset(self):
        self.output_tensor = None
        self.attn_weights = None

forensics = AttentionForensics()

# ==============================================================================
# -- Image Processing & Visualization Helpers ----------------------------------
# ==============================================================================
def preprocess_saved_image(img_path):
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    
    rgb_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_image)
    image = image.resize((g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])).convert('RGB')
    
    tensor_image = TF.to_tensor(image)
    tensor_image = TF.normalize(tensor_image, 
                                mean=g_conf.IMG_NORMALIZATION['mean'], 
                                std=g_conf.IMG_NORMALIZATION['std'])
    return tensor_image, rgb_image

def generate_master_panel(raw_images, global_attn, gradcam, fusion, heads_masks, head_weights, telemetry_data):
    """Generates a Split Vertical Widescreen Master Diagnostic Panel."""
    h, w, _ = raw_images[0].shape
    
    def apply_heatmap(img, mask):
        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return cv2.addWeighted(img_uint8, 0.5, heatmap, 0.5, 0)

    # Pre-process 3-camera horizontal rows
    row_raw = np.hstack(raw_images)
    row_g_attn = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, global_attn)])
    row_g_cam = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, gradcam)])
    row_fuse = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, fusion)])
    row_h1 = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, heads_masks[0])])
    row_h2 = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, heads_masks[1])])
    row_h3 = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, heads_masks[2])])
    row_h4 = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, heads_masks[3])])

    # Create a thick black vertical border
    border_width = 15
    border = np.zeros((h, border_width, 3), dtype=np.uint8)

    # Build the Grid Rows (Left Side | Border | Right Side)
    r1 = np.hstack([row_raw, border, row_h1])
    r2 = np.hstack([row_g_attn, border, row_h2])
    r3 = np.hstack([row_g_cam, border, row_h3])
    r4 = np.hstack([row_fuse, border, row_h4])
    grid_canvas = np.vstack([r1, r2, r3, r4])

    # Banner Design
    banner_h = 80
    total_w = (w * 6) + border_width
    banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
    
    # Evasion Flag Logic
    is_evading = telemetry_data.get('evasion', 'false') == 'true'
    evasion_color = (0, 0, 255) if is_evading else (0, 255, 0) # Red if True, Green if False
    evasion_text = "TRUE" if is_evading else "FALSE"

    text_left = f"FRAME: {telemetry_data.get('frame_id', 'N/A')} | CMD: {telemetry_data.get('command', 'N/A')}"
    text_center = f"LANE EVASION: {evasion_text}"
    text_right = f"STEER: {telemetry_data.get('steer', 0.0):.4f} | SPEED: {telemetry_data.get('speed', 0.0):.2f} km/h"
    
    cv2.putText(banner, text_left, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, text_center, (int(total_w/2) - 150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, evasion_color, 2, cv2.LINE_AA)
    cv2.putText(banner, text_right, (total_w - 500, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Grid Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_x = 10
    right_x = (w * 3) + border_width + 10

    cv2.putText(grid_canvas, "1. RAW INPUTS", (left_x, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, "2. GLOBAL ATTENTION", (left_x, h + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, "3. GRAD-CAM (STEER)", (left_x, (2*h) + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, "4. FUSION", (left_x, (3*h) + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(grid_canvas, f"5. HEAD 1 ({head_weights[0]:.1f}%)", (right_x, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, f"6. HEAD 2 ({head_weights[1]:.1f}%)", (right_x, h + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, f"7. HEAD 3 ({head_weights[2]:.1f}%)", (right_x, (2*h) + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, f"8. HEAD 4 ({head_weights[3]:.1f}%)", (right_x, (3*h) + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, grid_canvas])

# ==============================================================================
# -- Main Processor ------------------------------------------------------------
# ==============================================================================
def run_forensic_extraction(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Master XAI Extraction for Run: {args.run_name} on {device}...")

    conf_file_path = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\_results\Ours\Town12346_5\config40.json"
    exp_dir = os.path.dirname(conf_file_path)

    with open(conf_file_path, 'r') as f:
        configuration_dict = json.loads(f.read())
        
    g_conf.immutable(False)
    merge_with_yaml(os.path.join(exp_dir, configuration_dict['yaml']), process_type='drive')
    set_type_of_process('drive', root=r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main")

    model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', f"{model.name}_{configuration_dict['checkpoint']}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
    else: model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    target_layers = [model._model.encoder_embedding_perception.layer4[-1]]
    cam_extractor = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # --- MONKEY PATCH & HOOKS ---
    attn_module = model._model.tx_encoder.layers[-1].self_attn
    original_forward = attn_module.forward

    import functools
    @functools.wraps(original_forward)
    def custom_forward(*args, **kwargs):
        kwargs['average_attn_weights'] = False
        return original_forward(*args, **kwargs)

    attn_module.forward = custom_forward
    attn_module.register_forward_hook(forensics.hook)

    # --- TARGET RUN SETUP ---
    run_dir = os.path.join("test_results", args.run_name)
    telemetry_path = os.path.join(run_dir, "telemetry.csv")
    
    if not os.path.exists(telemetry_path):
        print(f"Error: Could not find telemetry at {telemetry_path}")
        sys.exit(1)

    master_dir = os.path.join("analysis", "master_results", args.run_name)
    os.makedirs(master_dir, exist_ok=True)

    df = pd.read_csv(telemetry_path)
    clean_column = df['Lane_Invasion'].astype(str).str.strip().str.lower()
    
    # --- FRAME DECISION LOGIC ---
    if args.all_frames:
        valid_indices = df.index.tolist()
        print("Mode: ALL FRAMES. Processing entire run...")
    elif args.start_frame is not None and args.end_frame is not None:
        frame_mask = (df['Frame'] >= args.start_frame) & (df['Frame'] <= args.end_frame)
        valid_indices = df[frame_mask].index.tolist()
        print(f"Mode: HARDCODED LIMITS. Processing frames {args.start_frame} to {args.end_frame}...")
    else:
        evasion_indices = df.index[clean_column == 'true'].tolist()
        frames_to_process = set()
        for idx in evasion_indices:
            frames_to_process.update(range(max(0, idx - 20), min(len(df), idx + 11)))
        valid_indices = sorted(list(frames_to_process))
        print("Mode: AUTOMATIC EVASION WINDOWS. Processing (-20 to +10) around invasions...")

    print(f"-> Extracted {len(valid_indices)} unique frames for {args.run_name}.")

    for idx in valid_indices:
        row = df.iloc[idx]
        frame_id = str(int(row['Frame']))
        cmd_idx = int(row['Command_Idx'])
        speed_ms = float(row['Speed_kmh']) / 3.6
        steer = float(row['Raw_Steer'])
        lane_invasion = clean_column.iloc[idx]

        left_path = os.path.join(run_dir, "left", f"{frame_id}.jpg")
        center_path = os.path.join(run_dir, "center", f"{frame_id}.jpg")
        right_path = os.path.join(run_dir, "right", f"{frame_id}.jpg")

        try:
            left_t, left_raw = preprocess_saved_image(left_path)
            center_t, center_raw = preprocess_saved_image(center_path)
            right_t, right_raw = preprocess_saved_image(right_path)
        except FileNotFoundError:
            continue

        left_t_req = left_t.unsqueeze(0).to(device).requires_grad_(True)
        center_t_req = center_t.unsqueeze(0).to(device).requires_grad_(True)
        right_t_req = right_t.unsqueeze(0).to(device).requires_grad_(True)
        
        input_tensors = [[ [left_t_req, center_t_req, right_t_req] ]]

        if g_conf.DATA_COMMAND_ONE_HOT:
            one_hot = np.zeros(g_conf.DATA_COMMAND_CLASS_NUM, dtype=np.float32)
            if 0 <= (cmd_idx - 1) < g_conf.DATA_COMMAND_CLASS_NUM: one_hot[cmd_idx - 1] = 1.0
            input_tensors.append([torch.from_numpy(one_hot).unsqueeze(0).to(device)])
        else:
            input_tensors.append([torch.tensor([[cmd_idx - 1]], dtype=torch.long).to(device)])

        normalized_speed = float(np.clip(abs(speed_ms - g_conf.DATA_NORMALIZATION['speed'][0]) / (g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0]), 0.0, 1.0))
        input_tensors.append([torch.tensor([[normalized_speed]], dtype=torch.float32).to(device)])

        # --- FORWARD AND BACKWARD PASS ---
        forensics.reset()
        with torch.set_grad_enabled(True):
            grayscale_cams = cam_extractor(input_tensor_list=input_tensors)
            
        gradcam_masks = [np.clip(cam, 0, 1) for cam in grayscale_cams]

        # --- LRP ATTRIBUTION MATH ---
        head_weights = [25.0, 25.0, 25.0, 25.0]
        if forensics.output_tensor is not None and forensics.output_tensor.grad is not None:
            nabla_Y = np.squeeze(forensics.output_tensor.grad.detach().cpu().numpy())
            if nabla_Y.ndim == 1: nabla_Y = np.expand_dims(nabla_Y, axis=0)
            
            W_O = model._model.tx_encoder.layers[-1].self_attn.out_proj.weight.detach().cpu().numpy() 
            nabla_X = np.matmul(nabla_Y, W_O) 
            
            raw_weights = [np.abs(nabla_X[:, i*128 : (i+1)*128]).sum() for i in range(4)]
            if sum(raw_weights) > 0:
                head_weights = [(w / sum(raw_weights)) * 100 for w in raw_weights]

        # --- PROCESS HEATMAPS ---
        feat_h, feat_w = model._model.res_out_h, model._model.res_out_w
        tokens_per_cam = feat_h * feat_w
        total_spatial_tokens = tokens_per_cam * 3
        target_size = (left_raw.shape[1], left_raw.shape[0])
        
        heads_masks_list = []
        for head_idx in range(4):
            head_attn = forensics.attn_weights[0, head_idx].detach().cpu().numpy().mean(axis=0)
            spatial_tokens = head_attn[:total_spatial_tokens]
            cam_masks = []
            for i in range(3):
                cam_grid = spatial_tokens[i * tokens_per_cam : (i + 1) * tokens_per_cam].reshape((feat_h, feat_w))
                c_max, c_min = cam_grid.max(), cam_grid.min()
                if c_max > c_min: cam_grid = (cam_grid - c_min) / (c_max - c_min)
                cam_masks.append(cv2.resize(cam_grid, target_size, interpolation=cv2.INTER_CUBIC))
            heads_masks_list.append(cam_masks)

        global_attn_masks = []
        fusion_masks = []
        for cam_idx in range(3):
            fused_cam = np.zeros(target_size[::-1], dtype=np.float32)
            for head_idx in range(4):
                fused_cam += (heads_masks_list[head_idx][cam_idx] * (head_weights[head_idx] / 100.0))
            fmax, fmin = fused_cam.max(), fused_cam.min()
            if fmax > fmin: fused_cam = (fused_cam - fmin) / (fmax - fmin)
            global_attn_masks.append(fused_cam)
            
            # Create Fusion (Global Attn * Grad-CAM)
            fuse = fused_cam * gradcam_masks[cam_idx]
            umax, umin = fuse.max(), fuse.min()
            if umax > umin: fuse = (fuse - umin) / (umax - umin)
            fusion_masks.append(fuse)

        # --- COMPILE AND SAVE ---
        telemetry_data = {'frame_id': frame_id, 'command': cmd_idx, 'steer': steer, 'speed': speed_ms * 3.6, 'evasion': lane_invasion}
        raw_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in [left_raw, center_raw, right_raw]]

        master_panel = generate_master_panel(raw_bgr, global_attn_masks, gradcam_masks, fusion_masks, heads_masks_list, head_weights, telemetry_data)
        cv2.imwrite(os.path.join(master_dir, f"frame_{frame_id}_master.jpg"), master_panel)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Diagnostic Master Panels for CILv2.")
    parser.add_argument('--run_name', type=str, required=True, help="Target run folder (e.g., run_007_Town02_rt0_LHS_006)")
    parser.add_argument('--start_frame', type=int, default=None, help="Specific starting frame number")
    parser.add_argument('--end_frame', type=int, default=None, help="Specific ending frame number")
    parser.add_argument('--all_frames', action='store_true', help="Process every frame in the run")
    
    args = parser.parse_args()
    run_forensic_extraction(args)