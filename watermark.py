import cv2
import numpy as np
import pywt

#水印嵌入
def embed_watermark(original_img_path, watermark_text, output_path, alpha=0.1):
    # 读取原始图像
    original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    
    # 将水印文本转换为二进制
    watermark_binary = ''.join(format(ord(c), '08b') for c in watermark_text)
    
    # 进行DWT变换
    coeffs = pywt.dwt2(original_img, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # 在LL子带中嵌入水印
    rows, cols = LL.shape
    watermark_length = len(watermark_binary)
    
    if watermark_length > rows * cols:
        raise ValueError("水印信息过大，无法嵌入")
    
    # 嵌入水印
    watermarked_LL = LL.copy()
    index = 0
    for i in range(rows):
        for j in range(cols):
            if index < watermark_length:
                # 修改LL系数的LSB
                watermarked_LL[i,j] = LL[i,j] * (1 + alpha * (int(watermark_binary[index]) - 0.5))
                index += 1
            else:
                break
    
    # 逆DWT变换
    watermarked_coeffs = (watermarked_LL, (LH, HL, HH))
    watermarked_img = pywt.idwt2(watermarked_coeffs, 'haar')
    
    # 保存含水印图像
    watermarked_img = np.uint8(watermarked_img)  # 强制转换为CV_8U
    cv2.imwrite(output_path, watermarked_img)
    
    return watermarked_img

#水印提取
def extract_watermark(watermarked_img_path, original_img_path, watermark_length, alpha=0.3):
    # 读取含水印图像和原始图像
    watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    
    # 对两幅图像进行DWT变换
    coeffs_watermarked = pywt.dwt2(watermarked_img, 'haar')
    LL_w, _ = coeffs_watermarked
    
    coeffs_original = pywt.dwt2(original_img, 'haar')
    LL_o, _ = coeffs_original
    
    # 提取水印
    extracted_binary = ''
    rows, cols = LL_w.shape
    index = 0
    
    for i in range(rows):
        for j in range(cols):
            if index < watermark_length * 8:  # 每个字符8位
                # 比较系数差异
                diff = (LL_w[i,j] - LL_o[i,j]) / (alpha * LL_o[i,j]) + 0.5
                bit = '1' if diff > 0.5 else '0'
                extracted_binary += bit
                index += 1
            else:
                break
    
    # 将二进制转换为文本
    watermark_text = ''
    for i in range(0, len(extracted_binary), 8):
        byte = extracted_binary[i:i+8]
        watermark_text += chr(int(byte, 2))
    
    return watermark_text

#鲁棒性测试
def robustness_test(watermarked_img_path, test_type, params):
    img = cv2.imread(watermarked_img_path)
    
    if test_type == 'rotate':
        # 旋转测试
        angle = params.get('angle', 30)
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(img, M, (cols, rows))
    
    elif test_type == 'crop':
        # 裁剪测试
        x, y, w, h = params.get('crop_area', (10, 10, 200, 200))
        return img[y:y+h, x:x+w]
    
    elif test_type == 'contrast':
        # 对比度调整
        alpha = params.get('alpha', 1.5)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    
    elif test_type == 'noise':
        # 添加噪声
        mean = 0
        var = params.get('var', 10)
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, img.shape)
        noisy_img = img + gaussian
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    elif test_type == 'compress':
        # JPEG压缩
        quality = params.get('quality', 50)
        cv2.imwrite('temp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imread('temp.jpg')
    
    else:
        return img.copy()
    
#测试
if __name__=="__main__":
    # 嵌入水印
    original_img_path = 'lena_grey.jpg'
    watermark_text = "Hello,world!"
    output_path = 'watermarked.jpg'
    watermarked_img = embed_watermark(original_img_path, watermark_text, output_path)
    # 提取水印
    extracted_watermark = extract_watermark(output_path, original_img_path, len(watermark_text))
    print(f"提取的水印: {extracted_watermark}")
    # 鲁棒性测试
    # 旋转攻击
    rotated_img = robustness_test(output_path, 'rotate', {'angle': 15})
    cv2.imwrite('rotated.jpg', rotated_img)
    # 对比度调整
    contrast_img = robustness_test(output_path, 'contrast', {'alpha': 1.8})
    cv2.imwrite('contrast.jpg', contrast_img)
    # 提取受攻击图像中的水印
    extracted_from_rotated = extract_watermark('rotated.jpg', original_img_path, len(watermark_text))
    print(f"从旋转图像中提取的水印: {extracted_from_rotated}")
    
    extracted_from_contrast = extract_watermark('contrast.jpg', original_img_path, len(watermark_text))
    print(f"从对比度调整图像中提取的水印: {extracted_from_contrast}")