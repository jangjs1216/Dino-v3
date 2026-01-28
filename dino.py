import argparse
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np

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
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = DinoSmallClassifier(NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Warning: 가중치 로드 실패 (랜덤 가중치로 실행됩니다). 에러: {e}")

    model.eval()
    return model


def predict_image(model, pil_image):
    input_tensor = inference_transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, 0)
        pred_class = CLASS_NAMES[top_idx.item()]
        confidence = top_prob.item() * 100

    return pred_class, confidence


def annotate_image(pil_image, text):
    # PIL로 텍스트 오버레이 (폰트 없을 경우 기본 폰트 사용)
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=28)
    except Exception:
        font = ImageFont.load_default()

    # 텍스트 위치 및 배경 박스
    margin = 10
    lines = text.split('\n')
    max_w = 0
    total_h = 0
    for line in lines:
        w, h = draw.textsize(line, font=font)
        max_w = max(max_w, w)
        total_h += h

    box_w = max_w + margin * 2
    box_h = total_h + margin * 2

    # 반투명 박스
    box = Image.new('RGBA', (box_w, box_h), (255, 255, 255, 200))
    img.paste(box, (10, 10), box)

    # 텍스트 그리기
    y = 10 + margin
    for line in lines:
        draw.text((10 + margin, y), line, fill='black', font=font)
        y += font.getsize(line)[1]

    return img


def save_side_by_side(original, annotated, out_path):
    w, h = original.size
    canvas = Image.new('RGB', (w * 2, h))
    canvas.paste(original, (0, 0))
    canvas.paste(annotated, (w, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def process_folder(model, folder_path, out_dir, show_each=False, max_images=None):
    png_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    if not png_paths:
        print(f"No PNG images found in {folder_path}")
        return

    for i, p in enumerate(png_paths):
        if max_images and i >= max_images:
            break

        img = Image.open(p).convert('RGB')
        pred_class, conf = predict_image(model, img)
        text = f"Pred: {pred_class}\nConf: {conf:.2f}%"
        annotated = annotate_image(img, text)

        base = os.path.basename(p)
        out_path = os.path.join(out_dir, base)
        save_side_by_side(img, annotated, out_path)
        print(f"Saved: {out_path}")

        if show_each:
            plt.figure(figsize=(8, 6))
            plt.imshow(np.array(Image.open(out_path)))
            plt.axis('off')
            plt.title(base)
            plt.show()

# ==========================================
# 실행
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run local inference with a DINO model and show original+result side-by-side.')
    parser.add_argument('--model', '-m', required=True, help='Path to local .pth/.pt model file')
    parser.add_argument('--folders', '-f', required=True, nargs='+', help='One or more folders containing 640x640 PNG images')
    parser.add_argument('--outdir', '-o', default='outputs', help='Directory to save side-by-side results')
    parser.add_argument('--show', action='store_true', help='Show each result with matplotlib')
    parser.add_argument('--max', type=int, default=None, help='Max images per folder to process')
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"Warning: 폴더가 아닙니다 -> {folder} (건너뜀)")
            continue

        folder_name = os.path.basename(os.path.normpath(folder))
        target_out = os.path.join(args.outdir, folder_name)
        print(f"Processing folder: {folder} -> {target_out}")
        process_folder(model, folder, target_out, show_each=args.show, max_images=args.max)

    print("모두 완료되었습니다. 출력 폴더를 확인하세요.")
