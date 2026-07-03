import os
import sys
import csv
import json
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import functools

# Import configurations
from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from _utils.grad_cam.grad_cam import GradCAM

class AttentionForensics:
    """An isolated vault to safely catch and guard tensors from Grad-CAM's clean-up passes."""
    def __init__(self):
        self.attn_weights = None
        self.output_tensor = None

    def hook(self, module, input, output):
        # Always update the weights so our Saliency panels draw the most recent visual frame
        self.attn_weights = output[1]
        
        # Immune System: Only capture the tensor if it actually has the autograd engine attached.
        # This prevents Grad-CAM's final 'no_grad' pass from overwriting our data!
        if output[0].requires_grad:
            self.output_tensor = output[0]
            self.output_tensor.retain_grad()
            
    def reset(self):
        self.output_tensor = None
        self.attn_weights = None

# Instantiate the vault
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

def generate_diagnostic_panel(raw_images, attention_masks, gradcam_masks, telemetry_data):
    """Row 1: Raw | Row 2: Dynamic Global Attention | Row 3: Grad-CAM | Row 4: Fusion"""
    h, w, _ = raw_images[0].shape
    
    fusion_masks = []
    for att, cam in zip(attention_masks, gradcam_masks):
        fused = att * cam
        fmax, fmin = fused.max(), fused.min()
        if fmax > fmin: fused = (fused - fmin) / (fmax - fmin)
        fusion_masks.append(fused)

    def apply_heatmap(img, mask):
        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return cv2.addWeighted(img_uint8, 0.5, heatmap, 0.5, 0)

    row_raw = np.hstack(raw_images)
    row_attn = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, attention_masks)])
    row_cam = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, gradcam_masks)])
    row_fuse = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, fusion_masks)])

    grid_canvas = np.vstack([row_raw, row_attn, row_cam, row_fuse])

    banner_h = 80
    total_w = w * 3
    banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
    text_left = f"FRAME: {telemetry_data.get('frame_id', 'N/A')} | CMD: {telemetry_data.get('command', 'N/A')}"
    text_right = f"STEER: {telemetry_data.get('steer', 0.0):.4f} | SPEED: {telemetry_data.get('speed', 0.0):.2f} km/h"
    cv2.putText(banner, text_left, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, text_right, (total_w - 450, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    font_scale, color, thickness = 0.6, (255, 255, 255), 1
    cv2.putText(grid_canvas, "1. RAW INPUTS (L / C / R)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(grid_canvas, "2. TRANSFORMER ATTENTION (DYNAMIC GRAD-WEIGHTED)", (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(grid_canvas, "3. GRAD-CAM (STEER SENSITIVITY)", (10, (2*h) + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(grid_canvas, "4. FUSION (SALIENCY * ATTENTION)", (10, (3*h) + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return np.vstack([banner, grid_canvas])

def generate_cognition_panel(raw_images, heads_masks_list, telemetry_data, head_weights):
    """5x3 canvas dedicated to the 4 independent Attention Heads with Live Weights."""
    h, w, _ = raw_images[0].shape
    
    def apply_heatmap(img, mask):
        img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return cv2.addWeighted(img_uint8, 0.5, heatmap, 0.5, 0)

    rows = [np.hstack(raw_images)]
    for head_idx, masks in enumerate(heads_masks_list):
        row = np.hstack([apply_heatmap(img, m) for img, m in zip(raw_images, masks)])
        rows.append(row)

    grid_canvas = np.vstack(rows)

    banner_h = 80
    total_w = w * 3
    banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
    text_left = f"FRAME: {telemetry_data.get('frame_id', 'N/A')} | CMD: {telemetry_data.get('command', 'N/A')}"
    text_right = f"STEER: {telemetry_data.get('steer', 0.0):.4f} | SPEED: {telemetry_data.get('speed', 0.0):.2f} km/h"
    cv2.putText(banner, text_left, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, text_right, (total_w - 450, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid_canvas, "1. RAW INPUTS", (10, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, f"2. HEAD 1 (Weight: {head_weights[0]:.1f}%)", (10, h + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, f"3. HEAD 2 (Weight: {head_weights[1]:.1f}%)", (10, (2*h) + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, f"4. HEAD 3 (Weight: {head_weights[2]:.1f}%)", (10, (3*h) + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_canvas, f"5. HEAD 4 (Weight: {head_weights[3]:.1f}%)", (10, (4*h) + 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, grid_canvas])

# ==============================================================================
# -- Main Batch Processor ------------------------------------------------------
# ==============================================================================
def run_forensic_extraction():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Dynamic Dual-Diagnostic Extraction on {device}...")

    conf_file_path = r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main\_results\Ours\Town12346_5\config40.json"
    exp_dir = os.path.dirname(conf_file_path)

    with open(conf_file_path, 'r') as f:
        configuration_dict = json.loads(f.read())
        
    g_conf.immutable(False)
    merge_with_yaml(os.path.join(exp_dir, configuration_dict['yaml']), process_type='drive')
    set_type_of_process('drive', root=r"C:\Users\sameh\Desktop\XAI\New folder\CILv2_multiview-main")

    print("Loading PyTorch weights...")
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

    @functools.wraps(original_forward)
    def custom_forward(*args, **kwargs):
        kwargs['average_attn_weights'] = False
        return original_forward(*args, **kwargs)

    attn_module.forward = custom_forward
    attn_module.register_forward_hook(forensics.hook)

    outliers_df = pd.read_csv("analysis/outlier_runs_table.csv", index_col="Run_Name")
    evasion_runs = outliers_df[outliers_df['Total_Evasions'] > 0].index.tolist()
    
    print(f"Found {len(evasion_runs)} runs. Commencing temporal batch extraction...")

    for run_name in evasion_runs:
        print(f"\nProcessing Run: {run_name}")
        run_dir = os.path.join("test_results", run_name)
        telemetry_path = os.path.join(run_dir, "telemetry.csv")
        
        cog_dir = os.path.join("analysis", "cognition_results", run_name)
        diag_dir = os.path.join("analysis", "diagnostic_results", run_name)
        os.makedirs(cog_dir, exist_ok=True)
        os.makedirs(diag_dir, exist_ok=True)

        df = pd.read_csv(telemetry_path)
        clean_column = df['Lane_Invasion'].astype(str).str.strip().str.lower()
        evasion_indices = df.index[clean_column == 'true'].tolist()
        
        frames_to_process = set()
        for idx in evasion_indices:
            frames_to_process.update(range(max(0, idx - 20), min(len(df), idx + 11)))
            
        valid_indices = sorted(list(frames_to_process))
        print(f"  -> Extracted {len(valid_indices)} unique frames for this run.")

        for idx in valid_indices:
            row = df.iloc[idx]
            frame_id = str(int(row['Frame']))
            cmd_idx = int(row['Command_Idx'])
            speed_ms = float(row['Speed_kmh']) / 3.6
            steer = float(row['Raw_Steer'])

            left_path = os.path.join(run_dir, "left", f"{frame_id}.jpg")
            center_path = os.path.join(run_dir, "center", f"{frame_id}.jpg")
            right_path = os.path.join(run_dir, "right", f"{frame_id}.jpg")

            try:
                left_t, left_raw = preprocess_saved_image(left_path)
                center_t, center_raw = preprocess_saved_image(center_path)
                right_t, right_raw = preprocess_saved_image(right_path)
            except FileNotFoundError:
                continue

            # --- FORCE AUTOGRAD GRAPH TO BUILD ---
            left_t_req = left_t.unsqueeze(0).to(device)
            left_t_req.requires_grad = True
            center_t_req = center_t.unsqueeze(0).to(device)
            center_t_req.requires_grad = True
            right_t_req = right_t.unsqueeze(0).to(device)
            right_t_req.requires_grad = True
            
            processed_cams = [left_t_req, center_t_req, right_t_req]
            formatted_imgs = [processed_cams]

            if g_conf.DATA_COMMAND_ONE_HOT:
                one_hot = np.zeros(g_conf.DATA_COMMAND_CLASS_NUM, dtype=np.float32)
                if 0 <= (cmd_idx - 1) < g_conf.DATA_COMMAND_CLASS_NUM: one_hot[cmd_idx - 1] = 1.0
                formatted_cmd = [torch.from_numpy(one_hot).unsqueeze(0).to(device)]
            else:
                formatted_cmd = [torch.tensor([[cmd_idx - 1]], dtype=torch.long).to(device)]

            normalized_speed = float(np.clip(abs(speed_ms - g_conf.DATA_NORMALIZATION['speed'][0]) / (g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0]), 0.0, 1.0))
            formatted_speed = [torch.tensor([[normalized_speed]], dtype=torch.float32).to(device)]
            input_tensors = [formatted_imgs, formatted_cmd, formatted_speed]

            # --- FORWARD AND BACKWARD PASS ---
            forensics.reset() # Wipe the vault clean for the new frame
            
            with torch.set_grad_enabled(True):
                grayscale_cams = cam_extractor(input_tensor_list=input_tensors)
                
            gradcam_masks = [np.clip(cam, 0, 1) for cam in grayscale_cams]

            # --- MANUAL LRP PROJECTION (DEFEATING THE GHOST TENSOR) ---
            head_weights = [25.0, 25.0, 25.0, 25.0]
            
            # Check the vault!
            if forensics.output_tensor is not None and forensics.output_tensor.grad is not None:
                # We successfully caught the gradient! Extract it:
                nabla_Y = forensics.output_tensor.grad.detach().cpu().numpy()
                nabla_Y = np.squeeze(nabla_Y) # Safely drop empty batch dimensions
                
                if nabla_Y.ndim == 1:
                    nabla_Y = np.expand_dims(nabla_Y, axis=0)
                
                # Extract the learned Projection Matrix
                W_O = model._model.tx_encoder.layers[-1].self_attn.out_proj.weight.detach().cpu().numpy() 
                
                # Manual LRP un-blending math
                nabla_X = np.matmul(nabla_Y, W_O) 
                
                raw_weights = []
                for i in range(4):
                    head_chunk_grad = nabla_X[:, i*128 : (i+1)*128]
                    chunk_importance = np.abs(head_chunk_grad).sum()
                    raw_weights.append(chunk_importance)
                
                total_importance = sum(raw_weights)
                if total_importance > 0:
                    head_weights = [(w / total_importance) * 100 for w in raw_weights]
            else:
                print(f"  [DEBUG] Frame {frame_id}: Vault failed to capture gradients.")

            # --- PROCESS THE 4 INDEPENDENT HEADS ---
            feat_h, feat_w = model._model.res_out_h, model._model.res_out_w
            tokens_per_cam = feat_h * feat_w
            total_spatial_tokens = tokens_per_cam * 3
            target_size = (left_raw.shape[1], left_raw.shape[0])
            
            heads_masks_list = []
            for head_idx in range(4):
                # Pull the visual matrices out of the vault
                head_attn = forensics.attn_weights[0, head_idx].detach().cpu().numpy().mean(axis=0)
                spatial_tokens = head_attn[:total_spatial_tokens]
                
                cam_masks = []
                for i in range(3):
                    cam_grid = spatial_tokens[i * tokens_per_cam : (i + 1) * tokens_per_cam].reshape((feat_h, feat_w))
                    c_max, c_min = cam_grid.max(), cam_grid.min()
                    if c_max > c_min: cam_grid = (cam_grid - c_min) / (c_max - c_min)
                    cam_masks.append(cv2.resize(cam_grid, target_size, interpolation=cv2.INTER_CUBIC))
                heads_masks_list.append(cam_masks)

            # --- CALCULATE DYNAMIC GRAD-WEIGHTED GLOBAL ATTENTION ---
            global_attn_masks = []
            for cam_idx in range(3):
                fused_cam = np.zeros(target_size[::-1], dtype=np.float32)
                for head_idx in range(4):
                    # Blend the camera masks strictly based on their dynamic mathematical importance
                    fused_cam += (heads_masks_list[head_idx][cam_idx] * (head_weights[head_idx] / 100.0))
                
                fmax, fmin = fused_cam.max(), fused_cam.min()
                if fmax > fmin: fused_cam = (fused_cam - fmin) / (fmax - fmin)
                global_attn_masks.append(fused_cam)

            # --- COMPOSITE AND SAVE ---
            telemetry_data = {'frame_id': frame_id, 'command': cmd_idx, 'steer': steer, 'speed': speed_ms * 3.6}
            raw_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in [left_raw, center_raw, right_raw]]

            # 1. Save Saliency Panel
            diag_panel = generate_diagnostic_panel(raw_bgr, global_attn_masks, gradcam_masks, telemetry_data)
            cv2.imwrite(os.path.join(diag_dir, f"frame_{frame_id}_diagnostic.jpg"), diag_panel)

            # 2. Save Cognition Panel
            cog_panel = generate_cognition_panel(raw_bgr, heads_masks_list, telemetry_data, head_weights)
            cv2.imwrite(os.path.join(cog_dir, f"frame_{frame_id}_cognition.jpg"), cog_panel)
    
    print("\nExtraction complete. All Diagnostic and Cognition panels successfully saved.")

if __name__ == '__main__':
    run_forensic_extraction()