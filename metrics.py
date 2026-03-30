import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def detect_edges(image, low, high):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)  # 30 50
    return edges

def edge_overlap(edge1, edge2):
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 0.0
    return intersection / union

    
def compute_siou(pred, target, left):
    left_edges = detect_edges(left, 100, 200)
    pred_edges = detect_edges(pred, 100, 200) # 5, 50
    right_edges = detect_edges(target, 100, 200) # 20 100


    diff_gl = abs(pred - left)
    diff_rl = abs(target - left)
    diff_gl = cv2.cvtColor(diff_gl, cv2.COLOR_BGR2GRAY)
    diff_rl = cv2.cvtColor(diff_rl, cv2.COLOR_BGR2GRAY)
    diff_gl_ = np.zeros(diff_rl.shape)
    diff_rl_ = np.zeros(diff_rl.shape)
    diff_gl_[diff_gl>5] = 1 # 5
    diff_rl_[diff_rl>5] = 1

    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    diff_overlap_grl =edge_overlap(diff_gl_, diff_rl_)


    return 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl

def eval_stereo(pred, target, left):
    max_pixel = 255.0
    assert pred.shape == target.shape
    diff = pred - target

    mse_err = np.mean(diff ** 2)

    rmse = np.sqrt(mse_err)

    absolute_errors = np.abs(diff)
    mae = np.mean(absolute_errors)
    
    # 对于完全相同的两张图像，将其 PSNR 直接设定为理想极大值32
    if rmse < 1e-8:
        psnr = 32.0
    else:
        psnr = 20 * np.log10(max_pixel / rmse)

    ssim_value, _ = ssim(pred, target, full=True, multichannel=True, win_size=7, channel_axis=2)
    siou_value = compute_siou(pred, target, left)

    return {'rmse': float(rmse), 'mse': float(mse_err), 'siou': float(siou_value), 'psnr': float(psnr), 'ssim': float(ssim_value)}
