import cv2
import numpy as np
import random
from PIL import Image


image = Image.open('C:/Users/alstj/Desktop/image1.jpg')
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def apply_augmentation(image_cv):
    augmented_image = image_cv.copy()

    # 가우시안 블러
    if random.random() < 0.5:
        blur_limit = random.randint(3, 7)
        # blur_limit이 짝수일 경우 홀수로 변경
        if blur_limit % 2 == 0:
            blur_limit += 1
        augmented_image = cv2.GaussianBlur(augmented_image, (blur_limit, blur_limit), 0)

    # 가우시안 노이즈
    if random.random() < 0.5:
        noise_var = random.uniform(10.0, 50.0)
        gauss_noise = np.random.normal(0, noise_var, augmented_image.shape).astype(np.uint8)
        augmented_image = cv2.add(augmented_image, gauss_noise)


    if random.random() < 0.7:
        rows, cols, _ = augmented_image.shape
        shift_x = random.uniform(-0.1, 0.1) * cols
        shift_y = random.uniform(-0.1, 0.1) * rows
        scale = random.uniform(0.9, 1.1)
        rotation = random.randint(-15, 15)

        # 이동 행렬
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented_image = cv2.warpAffine(augmented_image, M, (cols, rows))

        # 회전 및 크기 조정
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, scale)
        augmented_image = cv2.warpAffine(augmented_image, M, (cols, rows))

    if random.random() < 0.5:
        distort_limit = random.uniform(0.05, 0.1)
        augmented_image = cv2.resize(augmented_image, None, fx=1 + distort_limit, fy=1 + distort_limit)

    # 밝기 및 대비 변화
    if random.random() < 0.5:
        brightness = random.uniform(-0.2, 0.2)
        contrast = random.uniform(-0.2, 0.2)
        augmented_image = cv2.convertScaleAbs(augmented_image, alpha=1 + contrast, beta=brightness * 255)

    return augmented_image


# 증강된 이미지 생성 및 저장
augmented_images = []
num_samples = 5  # 생성할 데이터 개수

for i in range(num_samples):
    augmented = apply_augmentation(image_cv)
    output_aug_path = f"C:/Users/alstj/Desktop/data/image1_aug_{i}.jpg"
    cv2.imwrite(output_aug_path, augmented)
    augmented_images.append(output_aug_path)

# 생성된 파일 목록 반환
augmented_images
