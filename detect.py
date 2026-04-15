import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image, ImageDraw
from fast_plate_ocr import LicensePlateRecognizer
import numpy as np
import os
import argparse

# ============================================================
# ✏️  CHANGE THESE
# ============================================================
MODEL_PATH = "D:/vaishnavi/CV_Project/results/fasterrcnn_best.pth"
TEST_DIR   = "D:/vaishnavi/CV_Project/Dataset/test"       # ← base folder
SAVE_DIR   = "D:/vaishnavi/CV_Project/results/inference"
# ============================================================

NUM_CLASSES       = 2
CONFIDENCE_THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Faster R-CNN
# -----------------------------
def load_model(model_path):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model   = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# -----------------------------
# Detect plates
# -----------------------------
def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(img_tensor)

    boxes  = predictions[0]['boxes'].cpu()
    scores = predictions[0]['scores'].cpu()

    keep   = scores >= CONFIDENCE_THRESH
    boxes  = boxes[keep]
    scores = scores[keep]

    return img, boxes, scores


# -----------------------------
# Crop plate from box
# -----------------------------
def crop_plate(img, box):
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    pad = 5
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.width,  x2 + pad)
    y2 = min(img.height, y2 + pad)
    return img.crop((x1, y1, x2, y2))


# -----------------------------
# Run OCR
# -----------------------------
def run_ocr(ocr_reader, plate_crop):
    plate_gray = plate_crop.convert("L")
    try:
        result     = ocr_reader.run(np.array(plate_gray))
        plate_text = result[0] if result else "UNREADABLE"
    except Exception:
        plate_text = "UNREADABLE"
    return plate_text


# -----------------------------
# Process single image
# -----------------------------
def process_image(model, ocr_reader, image_path):
    img, boxes, scores = predict(model, image_path)

    if len(boxes) == 0:
        print(f"  ⚠️  No plates detected!")
        return

    draw = ImageDraw.Draw(img)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        plate_crop = crop_plate(img, box)
        plate_text = run_ocr(ocr_reader, plate_crop)

        # Draw green box + plate text on image
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 15), f"{plate_text} ({score:.2f})", fill="green")

        print(f"  ✅ Plate {i+1}: '{plate_text}' | conf: {score:.2f}")

    # Save output image with same name
    img_name  = os.path.basename(image_path)
    save_path = os.path.join(SAVE_DIR, img_name)
    img.save(save_path)
    print(f"  💾 Saved → {save_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="License Plate Detection + OCR")
    parser.add_argument('--image', type=str, required=True,
                        help='Image filename (e.g. car1.jpg) or full path')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ✅ If just filename given, build full path automatically
    if os.path.isabs(args.image):
        image_path = args.image                          # full path given
    else:
        image_path = os.path.join(TEST_DIR, args.image) # just filename given

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        exit()

    print("Loading Faster R-CNN...")
    model = load_model(MODEL_PATH)
    print("Model loaded ✅")

    print("Loading OCR...")
    ocr_reader = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
    print("OCR loaded ✅\n")

    print(f"Processing: {image_path}\n")
    process_image(model, ocr_reader, image_path)

    print("\n✅ Done!")