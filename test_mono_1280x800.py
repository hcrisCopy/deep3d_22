import os
import sys
import time
import math
import argparse
import numpy as np
import cv2
import torch
import psutil
from tqdm import tqdm

from data import transform, impro
from metrics import eval_stereo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int, help="choose your device")
    parser.add_argument("--model", default='/root/autodl-tmp/Deep3D/export/deep3d_v1.0_640x360_cuda.pt', type=str, help="input model path")
    parser.add_argument("--data_dir", default='/root/autodl-tmp/Data/mono2stereo-test', type=str, help="test dataset directory")
    parser.add_argument("--out_dir", default='/root/autodl-tmp/Data/deep3d_outputs', type=str, help="output directory for predicted right views")
    parser.add_argument('--inv', action='store_true', help='reverse left and right views')
    return parser.parse_args()

def evaluate():
    args = get_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    net = torch.jit.load(args.model)
    net.eval()
    process = transform.PreProcess()
    
    if 'cuda' in args.model.lower() and torch.cuda.is_available():
        net.to(args.gpu_id).half()
        process.to(args.gpu_id).half()
    else:
        args.gpu_id = -1
        
    out_width = int(os.path.basename(args.model).split('_')[2].split('x')[0])
    out_height = int(os.path.basename(args.model).split('_')[2].split('x')[1])
    
    TARGET_W, TARGET_H = 1280, 800
    subsets = ['animation', 'complex', 'indoor', 'outdoor', 'simple']
    
    # Warm up
    print("Warming up...")
    dummy_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    dummy_tensor = torch.from_numpy(dummy_img)
    if args.gpu_id >= 0:
        dummy_tensor = dummy_tensor.to(args.gpu_id).half()
    x_dummy = process(dummy_tensor)
    dummy_input = torch.cat((x_dummy, x_dummy, x_dummy, x_dummy, x_dummy, x_dummy), dim=0)
    dummy_input = dummy_input.reshape(1, *dummy_input.shape)
    with torch.no_grad():
        _ = net(dummy_input)
    if args.gpu_id >= 0:
        torch.cuda.synchronize()

    total_metrics = {'rmse': 0, 'mse': 0, 'siou': 0, 'psnr': 0, 'ssim': 0}
    total_samples = 0
    total_time = 0.0
    
    max_ram_usage = 0
    max_vram_allocated = 0
    max_vram_reserved = 0
    
    for subset in subsets:
        print(f"\nEvaluating subset: {subset}")
        subset_dir = os.path.join(args.data_dir, subset)
        left_dir = os.path.join(subset_dir, 'left')
        right_dir = os.path.join(subset_dir, 'right')
        
        if not os.path.exists(left_dir) or not os.path.exists(right_dir):
            print(f"Skipping {subset}, missing left or right folder.")
            continue
            
        filenames = sorted(os.listdir(left_dir))
        
        subset_metrics = {'rmse': 0, 'mse': 0, 'siou': 0, 'psnr': 0, 'ssim': 0}
        subset_valid_samples = 0
        
        for fname in tqdm(filenames, desc=subset):
            # Read images
            left_path = os.path.join(left_dir, fname)
            right_path = os.path.join(right_dir, fname)
            
            if not os.path.exists(right_path):
                continue
                
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            if left_img is None or right_img is None:
                continue
                
            # 1. Simulate 1280x800 input
            left_img_1280 = cv2.resize(left_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            right_img_1280 = cv2.resize(right_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            
            # 2. Resize to model's expected shape
            left_img_model = cv2.resize(left_img_1280, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Prepare tensor for single frame
            frame_tensor = torch.from_numpy(left_img_model)
            
            if args.gpu_id >= 0:
                frame_tensor = frame_tensor.to(args.gpu_id).half()
            
            x_processed = process(frame_tensor)
            
            # For a single image, temporal context is just the same image
            x1 = x_processed
            x2 = x_processed
            x0 = x_processed  # Initial state
            x3 = x_processed  # Current frame
            x4 = x_processed
            x5 = x_processed
            
            input_data = torch.cat((x1, x2, x0, x3, x4, x5), dim=0)
            input_data = input_data.reshape(1, *input_data.shape)
            
            # Inference
            if args.gpu_id >= 0:
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                out = net(input_data)
            
            if args.gpu_id >= 0:
                torch.cuda.synchronize()
                
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Update Memory Tracking
            current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            max_ram_usage = max(max_ram_usage, current_ram)
            if args.gpu_id >= 0:
                current_vram_allocated = torch.cuda.max_memory_allocated(args.gpu_id) / (1024 ** 2)
                current_vram_reserved = torch.cuda.max_memory_reserved(args.gpu_id) / (1024 ** 2)
                max_vram_allocated = max(max_vram_allocated, current_vram_allocated)
                max_vram_reserved = max(max_vram_reserved, current_vram_reserved)
            
            # Extract right view prediction
            right_pred = out[0]
            
            # Extract right view using deep3d's tensor2im
            pred_right_img_model = transform.tensor2im(right_pred)
            
            # 3. Resize prediction back to 1280x800
            pred_right_img_1280 = cv2.resize(pred_right_img_model, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            
            if total_samples == 0:
                print(f"\n\n--- 第一张图片尺寸变化追踪 ---")
                print(f"1. 原始图片大小: Left GT = {left_img.shape}, Right GT = {right_img.shape}")
                print(f"2. 模拟输入大小 (1280x800): Left = {left_img_1280.shape}, Right GT = {right_img_1280.shape}")
                print(f"3. 网络输入大小 (Left): {left_img_model.shape}")
                print(f"4. 网络输出大小 (Predicted Right): {pred_right_img_model.shape}")
                print(f"5. 计算指标使用的大小: Predicted Right = {pred_right_img_1280.shape}, Right GT = {right_img_1280.shape}, Left GT = {left_img_1280.shape}")
                print(f"------------------------------\n")

            # Save the predicted right view (saving the 1280x800 version)
            out_subset_dir = os.path.join(args.out_dir, subset)
            os.makedirs(out_subset_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_subset_dir, fname), pred_right_img_1280)
            
            metrics = eval_stereo(pred_right_img_1280, right_img_1280, left_img_1280)
            
            for k in metrics:
                subset_metrics[k] += metrics[k]
                total_metrics[k] += metrics[k]
                
            total_samples += 1
            subset_valid_samples += 1
            
        print(f"Results for {subset}:")
        if subset_valid_samples > 0:
            for k in subset_metrics:
                print(f"  {k}: {subset_metrics[k]/subset_valid_samples:.4f}")
            
    print("\n" + "="*40)
    print("Overall Results:")
    if total_samples > 0:
        for k in total_metrics:
            print(f"  {k}: {total_metrics[k]/total_samples:.4f}")
        
        avg_fps = total_samples / total_time
        print(f"\nTotal Samples: {total_samples}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        
        print("\nMemory Usage:")
        print(f"  Peak RAM Usage: {max_ram_usage:.2f} MB")
        if args.gpu_id >= 0:
            print(f"  Peak VRAM (Allocated Tensors): {max_vram_allocated:.2f} MB")
            print(f"  Peak VRAM (PyTorch Reserved):  {max_vram_reserved:.2f} MB")

if __name__ == "__main__":
    evaluate()
