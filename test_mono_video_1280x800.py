import os
import sys
import time
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
    parser.add_argument("--left_dir", default='/root/autodl-tmp/Data/mono_train/left', type=str, help="input left views directory")
    parser.add_argument("--right_dir", default='/root/autodl-tmp/Data/mono_train/right', type=str, help="input right views directory")
    parser.add_argument("--out_dir", default='/root/autodl-tmp/Data/deep3d_video_outputs', type=str, help="output directory for predicted right views")
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
    
    TARGET_W, TARGET_H = 1920, 1080
    # TARGET_W, TARGET_H = 1280, 800
    
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
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    video_length = 1000 # 0 to 10249
    alpha = 5
    
    print("Initializing frame pool...")
    frames_pool = []
    
    # Read first 2*alpha+1 frames
    for i in range(alpha * 2 + 1):
        idx = min(i, video_length - 1)
        # Using 9-digit format, e.g., 000000000.jpg
        fname = f"{idx:09d}.jpg"
        left_path = os.path.join(args.left_dir, fname)
        left_img = cv2.imread(left_path)
        if left_img is None:
            # Maybe the formatting is different, let's try reading and checking
            raise FileNotFoundError(f"Initial frame {left_path} not found.")
        
        # Simulate 1280x800 input
        left_img_1280 = cv2.resize(left_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
        left_img_model = cv2.resize(left_img_1280, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)
        
        frames_pool.append(torch.from_numpy(left_img_model))
        
    x0 = frames_pool[0]
    if args.gpu_id >= 0:
        x0 = x0.to(args.gpu_id).half()
    x0 = process(x0)
    
    print("Start video inference evaluating...")
    beta = 0
    for frame in tqdm(range(video_length), desc="Video Frames"):
        if frame < alpha:
            beta = 0
        elif alpha <= frame < video_length - alpha:
            beta = -(frame - alpha)

        if alpha < frame < video_length - alpha:
            read_idx = frame + alpha
            fname = f"{read_idx:09d}.jpg"
            left_path = os.path.join(args.left_dir, fname)
            left_img = cv2.imread(left_path)
            
            # If for some reason video is shorter, duplicate last frame
            if left_img is None:
                left_img = cv2.imread(os.path.join(args.left_dir, f"{read_idx-1:09d}.jpg"))
                
            left_img_1280 = cv2.resize(left_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            left_img_model = cv2.resize(left_img_1280, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)
                
            frames_pool.pop(0)
            frames_pool.append(torch.from_numpy(left_img_model))

        x1 = frames_pool[np.clip(frame - alpha + beta, 0, alpha * 2)]
        x2 = frames_pool[np.clip(frame - 1 + beta, 0, alpha * 2)]
        x3 = frames_pool[frame + beta]
        x4 = frames_pool[np.clip(frame + 1 + beta, 0, alpha * 2)]
        x5 = frames_pool[np.clip(frame + alpha + beta, 0, alpha * 2)]

        if args.gpu_id >= 0:
            x1, x2, x3, x4, x5 = x1.to(args.gpu_id).half(), x2.to(args.gpu_id).half(), x3.to(args.gpu_id).half(), x4.to(args.gpu_id).half(), x5.to(args.gpu_id).half()
        
        x1_p, x2_p, x3_p, x4_p, x5_p = process(x1), process(x2), process(x3), process(x4), process(x5)
        
        input_data = torch.cat((x1_p, x2_p, x0, x3_p, x4_p, x5_p), dim=0)
        input_data = input_data.reshape(1, *input_data.shape)
        
        if args.gpu_id >= 0:
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            out = net(input_data)
            x0 = out.clone().detach()[0]
            
        if args.gpu_id >= 0:
            torch.cuda.synchronize()
            
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Memory tracking
        current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        max_ram_usage = max(max_ram_usage, current_ram)
        if args.gpu_id >= 0:
            current_vram_allocated = torch.cuda.memory_allocated(args.gpu_id) / (1024 ** 2)
            current_vram_reserved = torch.cuda.memory_reserved(args.gpu_id) / (1024 ** 2)
            max_vram_allocated = max(max_vram_allocated, current_vram_allocated)
            max_vram_reserved = max(max_vram_reserved, current_vram_reserved)
            
        # Extract right view prediction
        right_pred = out[0]
        # Using exact logic:
        # right_pred = right_pred
        # Wait, the `impro` or `transform.tensor2im` expecting specific shape:
        pred_right_img_model = transform.tensor2im(right_pred)
        
        # 3. Resize prediction back to 1280x800
        pred_right_img_1280 = cv2.resize(pred_right_img_model, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
        
        # Read GT
        gt_fname = f"{frame:09d}.jpg"
        right_path = os.path.join(args.right_dir, gt_fname)
        right_img = cv2.imread(right_path)
        left_path_eval = os.path.join(args.left_dir, gt_fname)
        left_img_eval = cv2.imread(left_path_eval)
        
        if right_img is not None and left_img_eval is not None:
            right_img_1280 = cv2.resize(right_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            left_img_eval_1280 = cv2.resize(left_img_eval, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
            
            if total_samples == 0:
                print(f"\n\n--- 第一张图片尺寸变化追踪 ---")
                print(f"1. 原始图片大小: Left GT = {left_img_eval.shape}, Right GT = {right_img.shape}")
                print(f"2. 模拟输入大小 (1280x800): Left = {left_img_eval_1280.shape}, Right GT = {right_img_1280.shape}")
                print(f"3. 网络输入大小 (Left): {left_img_model.shape}")
                print(f"4. 网络输出大小 (Predicted Right): {pred_right_img_model.shape}")
                print(f"5. 计算指标使用的大小: Predicted Right = {pred_right_img_1280.shape}, Right GT = {right_img_1280.shape}, Left GT = {left_img_eval_1280.shape}")
                print(f"------------------------------\n")
            
            # Save the predicted right view (saving the 1280x800 version)
            cv2.imwrite(os.path.join(args.out_dir, gt_fname), pred_right_img_1280)
            
            # Evaluate
            metrics = eval_stereo(pred_right_img_1280, right_img_1280, left_img_eval_1280)
            
            for k in metrics:
                total_metrics[k] += metrics[k]
                
            total_samples += 1
        else:
            # If no GT, just save the predicted one
            cv2.imwrite(os.path.join(args.out_dir, gt_fname), pred_right_img_1280)

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
