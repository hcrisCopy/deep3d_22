import cv2
import os

def blend_images(img1_path, img2_path, output_path, alpha=0.5):
    if not os.path.exists(img1_path):
        print(f"Error: 找不到图片 {img1_path}")
        return
    if not os.path.exists(img2_path):
        print(f"Error: 找不到图片 {img2_path}")
        return

    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Error: 图片读取失败")
        return

    # 确保两张图片的尺寸一致，如果不一致则将图2缩放至图1尺寸
    if img1.shape != img2.shape:
        print(f"尺寸不匹配，正在将 {img2_path} 缩放以匹配原图")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 半透明叠加 (alpha 为 img1 的权重，1-alpha 为 img2 的权重)
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

    # 保存结果
    cv2.imwrite(output_path, blended)
    print(f"叠加完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    image1 = "/root/autodl-tmp/Data/deep3d_outputs/animation/000000826.jpg"
    image2 = "/root/autodl-tmp/Data/mono2stereo-test/animation/right/000000826.jpg"
    out_img = "/root/autodl-tmp/Deep3D/compare_result.jpg"
    
    blend_images(image1, image2, out_img, alpha=0.5)
