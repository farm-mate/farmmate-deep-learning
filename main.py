# main.py

from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'resnet50.pth'  # 여러분의 모델 파일 경로로 변경
NUM_CLASSES = 33  # 실제 클래스 수에 맞게 변경하세요

# Flask 앱 설정
app = Flask(__name__)

# 모델 로드
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)  # 모델의 마지막 레이어 교체
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()  # 평가 모드로 설정

# 이미지 전처리 함수
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0).to(DEVICE)

# 예측 함수
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    plant_type = request.form.get('plantType')  # 이 부분은 실제 모델에서 사용되지 않을 수 있습니다.

    if not file:
        return jsonify({'error': 'No image provided'}), 400

    # 이미지 파일을 읽고 전처리
    image_bytes = file.read()
    tensor = transform_image(image_bytes)

    # 이미지 예측
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    # 결과 반환
    return jsonify({'plantType': plant_type, 'predictedClass': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
