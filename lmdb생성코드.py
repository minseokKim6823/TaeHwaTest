import os
import lmdb
import cv2
import numpy as np

def check_image_is_valid(image_path):
    """이미지가 올바르게 로드될 수 있는지 확인하는 함수"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        return True
    except:
        return False

def writeCache(env, cache):
    """LMDB에 데이터 저장"""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def createDataset(input_path, output_path, map_size=10**9):
    """LMDB 데이터셋 생성"""
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=map_size)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cache = {}
    cnt = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue  # 빈 줄 스킵

        parts = line.split(maxsplit=1)  # 공백을 기준으로 1회만 split (첫 번째 값: 이미지 경로, 나머지: 라벨)
        if len(parts) < 2:
            print(f"⚠️ 잘못된 형식: {line}")
            continue

        img_path, label = parts  # 첫 번째 값이 이미지 경로, 두 번째 값이 라벨

        if not os.path.exists(img_path) or not check_image_is_valid(img_path):
            print(f"⚠️ 이미지를 찾을 수 없음: {img_path}")
            continue

        # 이미지 로드 및 인코딩 (바이너리 형태)
        img = cv2.imread(img_path)
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bin = img_encoded.tobytes()

        image_key = f'image-{cnt:09d}'
        label_key = f'label-{cnt:09d}'

        cache[image_key] = img_bin  # 이미지 데이터를 LMDB에 저장
        cache[label_key] = label.encode()  # 문자열을 바이너리로 변환하여 저장

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f'✅ {cnt}개의 데이터 변환 완료')

        cnt += 1

    writeCache(env, cache)

    with env.begin(write=True) as txn:
        txn.put("num-samples".encode(), str(cnt - 1).encode())

    print(f"🎉 총 {cnt - 1}개의 데이터가 변환되었습니다!")

if __name__ == "__main__":
    input_path = "C:/data/label.txt"  # label.txt 파일 (이미지 경로 + 라벨 정보)
    output_path = "C:/data/lmdb"  # LMDB 데이터셋 저장 경로

    createDataset(input_path, output_path)