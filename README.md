# 태화이노베이션 과제테스트(paddleOCR을 이용한 글자 인식)

## 🔹 세팅 방법

```bash
pip install paddlepaddle
```

```python
import paddle
print("PaddlePaddle 설치 완료!")
```

# 방법론1 - 사진을 최적화 해보자!

생각한 이유 - paddlepaddle을 사용해본 결과 **성능이 훌륭**했다.
따라서, 사진의 명암 대비를 확연하게 가져간다면 글자를 잘 구분할 수 있을 것이라고 판단을 했다.   

### 1. 첫 번째 실행 코드와 jpg 파일

```python
from paddleocr import PaddleOCR

# OCR 객체 생성 (한국어 설정)
ocr = PaddleOCR(lang="korean")

# 이미지 경로 설정
img_path = 'C:/Users/alstj/Desktop/hello.jpg'

# OCR 수행
result = ocr.ocr(img_path)

# 첫 번째 결과 가져오기
result = result[0]

# 인식된 텍스트만 출력
for line in result:
    print(line[1][0])  
```

![image](https://github.com/user-attachments/assets/a5ea18ff-5051-460f-9421-41bf5bf3cf85)<br>
**hello.jpg**
<br><br><br><br>
![image](https://github.com/user-attachments/assets/1f246389-f5f2-40b0-b688-d0c5a1c930a0)<br>
**실행결과**
<br><br>
‘안녕’이 아닌 ‘안형’이 나옴….

### 2. 두 번째 실행 코드와 jpg 파일(동일한 jpg 파일로 시도 했습니다.)

첫번째 시도가 실패하여 웹서핑을 통해 알아본 결과, 인식된 텍스트의 신뢰도 임계값이 낮아서 오인식이 발생할 가능성이 있다고 한다.

따라서 OCR 객체 생성시  drop_score=0.7을 추가하여 신뢰도가 0.7이상인 텍스트만 출력하도록 했다.

```python
from paddleocr import PaddleOCR

# OCR 객체 생성 (한국어 설정)
ocr = PaddleOCR(lang="korean", drop_score=0.7)

# 이미지 경로 설정
img_path = 'C:/Users/alstj/Desktop/hello.jpg'

# OCR 수행
result = ocr.ocr(img_path)

# 첫 번째 결과 가져오기
result = result[0]

# 인식된 텍스트만 출력
for line in result:
    print(line[1][0])  
```

![image](https://github.com/user-attachments/assets/c4130cb9-bcf6-4c8a-bd77-1618a11a02bf)


안타깝게도 결과값은 변화가 없었다.😂

### 3. 세 번째 시도 : **GrayScale 적용하기**

PaddleOCR에서 **GrayScale**을 추가해서 OCR 성능을 최적화 해보았다.

<aside>
💡

### GrayScale이란?

**Grayscale**(그레이스케일)은 흑백 이미지를 의미한다.

즉, 색상(컬러) 정보를 제거하고 밝기(intensity)만 남긴 이미지.

ex. 흑백 TV, 스캐너로 스캔한 문서 

</aside>

![image](https://github.com/user-attachments/assets/334846d6-f152-4d55-b574-b32d79224d03)


```python
import cv2
from paddleocr import PaddleOCR

# OCR 객체 생성 (한국어 설정)
ocr = PaddleOCR(lang="korean", drop_score=0.7)

#이미지경로
img_path = 'C:/Users/alstj/Desktop/hello.jpg'

# 이미지를 그레이스케일로 변환
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 그레이스케일 이미지를 다시 저장
gray_img_path = 'C:/Users/alstj/Desktop/gray_hello.jpg'
cv2.imwrite(gray_img_path, img)

result = ocr.ocr(gray_img_path)

result = result[0]

for line in result:
    print(line[1][0])
```

 

![image](https://github.com/user-attachments/assets/f904e6bb-75b0-4780-b187-ebe1b964dc14)<br>
GrayScale 전
<br><br><br><br>
![image](https://github.com/user-attachments/assets/cda0127a-2160-44b5-a171-4f3c7ec64166)<br>
GrayScale 진행 후(파일명이 gray_hello임) 
<br><br><br><br>
![image](https://github.com/user-attachments/assets/06b00643-55fb-4405-9146-b7bb28e4f0e7)<br>
**GrayScale**을 반영해본 결과 안녕은 안영으로 결과가 나왔다

### 4. 네 번째 시도 : 바이너리 이미지로 변환하기

GrayScale에 대해 추가적으로 조사해본 결과,  검정색,회색,흰색 세 색깔중 하나로 전처리를 해주는 것을 깨달았고 흰색과 검정색만으로 구분지어 글자와 바탕을 확실하게 구분지을 수있도록 임계값을 조정해주는 작업을 해본 결과 

```python
import cv2
from paddleocr import PaddleOCR

# OCR 객체 생성 (한국어 설정)
ocr = PaddleOCR(lang="korean", drop_score=0.7)

# 이미지 경로 설정
img_path = 'C:/Users/alstj/Desktop/hello.jpg'

# 이미지를 그레이스케일로 불러오기
gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 흑백(바이너리) 이미지로 변환: 임계값 210 이하 -> 0, 초과 -> 255
_, binary_img = cv2.threshold(gray_img, 210, 255, cv2.THRESH_BINARY)

# 변환된 이미지를 임시 파일로 저장 (OCR은 파일 경로를 사용)
binary_img_path = 'C:/Users/alstj/Desktop/binary_hello.jpg'
cv2.imwrite(binary_img_path, binary_img)

# OCR 수행 (변환된 이미지 사용)
result = ocr.ocr(binary_img_path)

# 첫 번째 결과 가져오기
result = result[0]

# 인식된 텍스트만 출력
for line in result:
    print(line[1][0])
```

![image](https://github.com/user-attachments/assets/d8cf8956-9992-4dba-b9a8-6aea42f77aa7)<br>
**변환 결과**
<br><br><br><br><br><br><br>
![image](https://github.com/user-attachments/assets/5ba15a6d-8ef4-4e8a-a77f-43d5111c8f4c)<br>
**출력 결과**

                

성공적으로 값이 출력됐다!

# 방법론2 - 학습시키기

PaddleOCR을 gitHub로 부터 클론을 받고 커스터마이징을 했다.
GPU를 사용하기 않고 실행을 하기 때문에 학습 데이터에 따라서 학습시간이 달라진다.(epoch 50 학습데이터 150개 ⇒ 20시간 정도 걸림)

![image](https://github.com/user-attachments/assets/06994a42-dd92-4efb-ac07-c0baabc7613a)


**korean_PP-OCRv3_rec_train.yml**

```java
Global:
  use_gpu: false
  epoch_num: 50  # 학습할 총 Epoch 수
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_korean  # 모델 저장 경로
  save_epoch_step: 10  # 몇 번의 Epoch마다 모델을 저장할지
  eval_batch_step: [0, 5000]  # 평가할 Step
  cal_metric_during_train: True
  pretrained_model:
  checkpoints:
  save_inference_dir: ./inference/rec_korean  # 변환된 모델 저장 경로
  use_visualdl: False  # 학습 기록 시각화 여부
  infer_img: dataset/train_images/image1.jpg  # 테스트할 이미지 경로
  character_dict_path: ppocr/utils/dict/korean_dict.txt  #  한글 문자 사전 추가
  use_space_char: False

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001  # 학습률 (필요하면 조정)
  regularizer:
    name: L2
    factor: 0.00004

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet  #  학습 모델과 일치하게 유지
  Transform: null
  Backbone:
    name: SVTRNet
    img_size: [128, 640]  #  학습할 때 입력 크기 (추론할 때도 맞춰야 함)
    out_channels: 192
    layers: [3, 6, 3]
    hidden_dim: 192
    num_heads: [4, 8, 8]
    mixer: ["Local", "Local", "Local", "Global", "Global", "Global", "Global", "Global", "Global", "Global", "Global", "Global"]
    use_guide: True
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
  Head:
    name: CTCHead
    fc_decay: 4.0e-05

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode
Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: LMDBDataSet
    data_dir: C:/data/lmdb/  # 생성한 LMDB 데이터셋 경로
    label_file_path: C:/data/label.txt  # 라벨 파일 경로
    transforms:
      - DecodeImage:
          img_mode: BGR
      - CTCLabelEncode:
          max_text_length: 25
      - RecResizeImg:
          image_shape: [3, 128, 640]  # 학습할 때 크기 (추론할 때도 동일해야 함)
      - KeepKeys:
          keep_keys: ['image', 'label']
  loader:
    shuffle: True
    batch_size_per_card: 4
    drop_last: False
    num_workers: 4

Eval:
  dataset:
    name: LMDBDataSet
    data_dir: C:/Users/alstj/Desktop/TaeHwa/PaddleOCR/dataset/val_lmdb # 테스트용 LMDB 데이터셋 경로
    label_file_path: C:/Users/alstj/Desktop/TaeHwa/PaddleOCR/dataset/val.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
      - CTCLabelEncode:
          max_text_length: 25
      - RecResizeImg:
          image_shape: [3, 128, 640]  #  학습할 때 크기와 일치하게 수정
      - KeepKeys:
          keep_keys: ['image', 'label']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 16
    num_workers: 4

```

**실행방법(학습시키기)**

```java
python tools/train.py -c configs/rec/korean_PP-OCRv3_rec_train.yml
```

처음에 학습 할때 64 X 256으로 리사이징 했더니 사진이 찌그러지는 이슈가 생겨서 

128X640 으로 바꿨다

**test.py**
```pytho
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


def resize_with_padding(image, target_size=(640, 128)):
    """
    원본 이미지 비율을 유지하면서 target_size (width, height)로 변환 (패딩 추가)
    """
    old_size = image.size  # (width, height)
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    img_resized = image.resize(new_size, Image.LANCZOS)

    new_img = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_size[0] - new_size[0]) // 2
    paste_y = (target_size[1] - new_size[1]) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img


def enhance_image(image):
    """
    이미지 대비 증가 및 선명화 (적응형 이진화를 사용)
    """
    img = np.array(image)
    # 1️⃣ 대비 증가: CLAHE 적용 (clipLimit=3.5)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 2️⃣ 텍스트 강조: 적응형 이진화
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # OTSU 이진화로 임계값 결정
    _, img_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 배경 반전: 일반적으로 흰 배경, 검은 글씨를 가정
    img_bin = cv2.bitwise_not(img_bin)

    return Image.fromarray(img_bin)


# 학습된 모델에 맞게 OCR 설정 (학습 config에 따라 rec_image_shape="3,128,640", max_text_length:50 등으로 학습했다고 가정)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='korean',
    det_model_dir='./output/det_korean',
    rec_model_dir='./output/rec_korean/latest',  # 최신 학습된 모델 사용 (Inference 모델 변환 후 사용 가능)
    rec_algorithm='SVTR_LCNet',
    rec_image_shape="3,128,640",  # 학습 설정과 동일하게
    rec_char_dict_path="ppocr/utils/dict/korean_dict.txt",
    use_space_char=True,
    drop_score=0.0001,
    pretrained_model="",  # pretrained_model 옵션은 inference 모델 변환 후 필요하지 않으면 비워두기
)

# 원본 이미지 불러오기
img_path = "dataset/train_images/image1.jpg"  # 학습 config의 infer_img 경로
img = Image.open(img_path)

# 필요시 작은 이미지라면 업스케일 (여기서는 그대로 진행)
if img.size[0] < 640 or img.size[1] < 128:
    scale = max(640 // img.size[0], 128 // img.size[1])
    img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.LANCZOS)
    print(f"Image upscaled to: {img.size}")

# 이미지 크기를 640x128로 맞추기 (패딩 적용)
img_resized = resize_with_padding(img, target_size=(640, 128))
# 전처리: 대비 강화 및 이진화 적용
img_processed = enhance_image(img_resized)

# 저장 (디버깅용)
enhanced_img_path = "dataset/train_images/enhanced_image1.jpg"
img_processed.save(enhanced_img_path)
print(f"Processed image saved at: {enhanced_img_path}")

# OCR 실행
result = ocr.ocr(enhanced_img_path, cls=True)

# OCR 결과 확인
if result and result[0]:
    print("Detected Text Results:")
    for line in result[0]:
        print(f"Text: {line[1][0]} | Confidence: {line[1][1]}")

    # 감지된 박스 시각화 (OpenCV 이미지 사용)
    img_cv = cv2.imread(enhanced_img_path)
    boxes = [entry[0] for entry in result[0]]
    txts = [entry[1][0] for entry in result[0]]
    scores = [entry[1][1] for entry in result[0]]
    result_img = draw_ocr(img_cv, boxes, txts, scores, font_path="ppocr/utils/dict/korean_dict.txt")
    result_img = Image.fromarray(result_img)
    debug_image_path = "output/result_image.jpg"
    result_img.save(debug_image_path)
    print(f"Result image saved at: {debug_image_path}")
else:
    print("No text detected!")
```
