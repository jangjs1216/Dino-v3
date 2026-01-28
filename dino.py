import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# 학습된 모델 가중치 파일 경로 (.pth 또는 .pt)
MODEL_PATH = 'smartphone_classifier_small.pth' 

# 테스트할 이미지 경로
TEST_IMAGE_PATH = 'test_image.jpg' 

# 학습할 때 사용했던 클래스 이름들 (순서 중요!)
CLASS_NAMES = ['galaxy_s24', 'iphone_15_pro', 'z_flip_5', 'pixel_8', 'nothing_phone'] 
NUM_CLASSES = len(CLASS_NAMES)

# 사용 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 모델 구조 정의 (ViT-Small/14 Distilled)
# ==========================================
class DinoSmallClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DinoSmallClassifier, self).__init__()
        
        # ---------------------------------------------------------
        # [핵심 변경] ViT-S (Small) 모델 로드
        # Meta의 DINOv2 repo에서 'dinov2_vits14'를 가져옵니다.
        # (이 모델은 거대 모델로부터 지식을 증류(Distilled) 받은 모델입니다)
        # ---------------------------------------------------------
        print("Loading DINO ViT-Small backbone...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # 백본의 출력 차원 자동 확인 (Small은 보통 384)
        embed_dim = self.backbone.embed_dim 
        print(f"Embedding Dimension: {embed_dim}") # 384가 출력되어야 정상
        
        # 분류기 (Head) 정의
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# ==========================================
# 3. 전처리 정의 (640x640)
# ==========================================
inference_transform = transforms.Compose([
    transforms.Resize((640, 640)), # 640x640 해상도 강제 적용
    transforms.ToTensor(),
    # ImageNet 표준 정규화 (DINO 학습 시 필수)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. 추론 및 시각화 함수
# ==========================================
def predict_and_visualize(img_path, model_path):
    # --- A. 모델 초기화 및 가중치 로드 ---
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일이 없습니다 -> {model_path}")
        # (테스트를 위해 가중치 로드 없이 진행하려면 아래 줄 주석 처리)
        return

    model = DinoSmallClassifier(NUM_CLASSES).to(DEVICE)
    
    # 학습된 가중치 덮어쓰기
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Warning: 가중치 로드 실패 (랜덤 가중치로 실행됩니다). \n에러: {e}")

    model.eval() # 평가 모드 (Dropout 해제)

    # --- B. 이미지 로드 ---
    if not os.path.exists(img_path):
        print(f"Error: 이미지를 찾을 수 없습니다 -> {img_path}")
        return

    original_img = Image.open(img_path).convert('RGB')
    
    # 전처리 및 배치 차원 추가 (3, 640, 640) -> (1, 3, 640, 640)
    input_tensor = inference_transform(original_img).unsqueeze(0).to(DEVICE)

    # --- C. 추론 진행 ---
    print(f"Inferencing with resolution 640x640 on {DEVICE}...")
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Top 1 예측 결과
        top_prob, top_idx = torch.max(probabilities, 0)
        pred_class = CLASS_NAMES[top_idx.item()]
        confidence = top_prob.item() * 100

    # --- D. 시각화 (Matplotlib) ---
    plt.figure(figsize=(12, 8))
    
    # 원본 이미지 표시
    plt.imshow(original_img)
    plt.axis('off')
    
    # 결과 텍스트 오버레이
    result_text = f"Pred: {pred_class}\nConf: {confidence:.2f}%"
    
    # 텍스트 박스 스타일 설정
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    
    # 이미지 좌측 상단에 결과 표시
    plt.text(10, 50, result_text, fontsize=20, color='darkblue', 
             fontweight='bold', bbox=bbox_props, verticalalignment='top')

    # 타이틀
    plt.title(f"DINO ViT-S/16 (Distilled) Inference Result", fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n>>> 최종 예측: {pred_class} ({confidence:.2f}%)")

# ==========================================
# 실행
# ==========================================
if __name__ == '__main__':
    # 1. 먼저 테스트용 가짜 모델 파일을 만들거나, 실제 파일 경로를 지정하세요.
    # predict_and_visualize(TEST_IMAGE_PATH, MODEL_PATH)
    
    # [참고] 모델 파일이 없을 때 테스트를 위해 모델만 로드해보는 코드:
    print("모델 다운로드 및 구조 테스트 중...")
    temp_model = DinoSmallClassifier(NUM_CLASSES)
    print("성공! 위 구조대로 학습된 .pth 파일만 있으면 됩니다.")
