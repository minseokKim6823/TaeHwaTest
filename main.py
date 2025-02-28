from paddleocr import PaddleOCR

# OCR 객체 생성 (한국어 설정)
ocr = PaddleOCR(lang="korean")

# 이미지 경로 설정
img_path = 'C:/Users/alstj/Desktop/hello2.jpg'

# OCR 수행
result = ocr.ocr(img_path)

# 첫 번째 결과 가져오기
result = result[0]

# 인식된 텍스트만 출력
for line in result:
    print(line[1][0])