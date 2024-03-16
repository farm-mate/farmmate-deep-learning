from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = 'resnet50.pth'  # 기본 모델 파일 경로
NUM_CLASSES = 3  # 실제 클래스 수에 맞게 변경하세요

app = Flask(__name__)  # Flask 앱 설정

# 이미지 전처리 함수
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # RGB로 변환
    return my_transforms(image).unsqueeze(0).to(DEVICE)

# 예측 함수
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    plant_type = request.form.get('plantType', '')  # 기본값 설정

    # 식물 유형에 따라 모델 경로와 클래스 수 설정
    if plant_type:
        model_path = f'{plant_type}_resnet50.pth'
        num_classes = {
            '쥬키니호박' : 2,
            '토마토': 3,
            '애호박': 2,
            '단호박' : 3,
            '포도' : 2,
            '참외' : 3,
            '오이' : 2,
            '수박' : 3,
            '상추' : 3,
            '딸기' : 3,
            '고추' : 3,
            '가지' : 3
        }.get(plant_type, 2)  # 기본값으로 2 설정
    else:
        model_path = DEFAULT_MODEL_PATH
        num_classes = NUM_CLASSES  # 기본 클래스 수 사용

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    if not file:
        return jsonify({'error': 'No image provided'}), 400

    image_bytes = file.read()
    tensor = transform_image(image_bytes)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    return jsonify({'plantType': plant_type, 'predictedClass': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
