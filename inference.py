"""YOLO26n ONNX推論スクリプト"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse

# COCOクラス名
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 色をクラスごとに生成
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def preprocess(image: np.ndarray, input_size: tuple = (640, 640)) -> tuple:
    """画像の前処理

    Args:
        image: 入力画像 (BGR)
        input_size: モデルの入力サイズ (H, W)

    Returns:
        blob: 前処理済みテンソル
        scale: スケール係数
        pad_w: 横方向パディング
        pad_h: 縦方向パディング
    """
    h, w = image.shape[:2]

    # アスペクト比を維持してリサイズ
    scale = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # パディング (グレー: 114)
    pad_h = (input_size[0] - new_h) // 2
    pad_w = (input_size[1] - new_w) // 2
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    # 正規化とチャンネル順変更
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, 0)  # バッチ次元追加

    return blob, scale, pad_w, pad_h


def postprocess(outputs: list, scale: float, pad_w: int, pad_h: int,
                conf_threshold: float = 0.25) -> list:
    """出力の後処理

    Args:
        outputs: モデル出力
        scale: 前処理時のスケール係数
        pad_w: 前処理時の横方向パディング
        pad_h: 前処理時の縦方向パディング
        conf_threshold: 信頼度閾値

    Returns:
        検出結果のリスト
    """
    detections = outputs[0][0]  # [300, 6]

    results = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det

        if score < conf_threshold:
            continue

        # パディングとスケールを元に戻す
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        results.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'score': float(score),
            'class_id': int(class_id)
        })

    return results


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """検出結果を描画

    Args:
        image: 入力画像
        detections: 検出結果リスト

    Returns:
        描画済み画像
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        class_id = det['class_id']

        # クラス名と色
        if 0 <= class_id < len(COCO_CLASSES):
            class_name = COCO_CLASSES[class_id]
            color = tuple(int(c) for c in COLORS[class_id])
        else:
            class_name = f"class_{class_id}"
            color = (0, 255, 0)

        # バウンディングボックス描画
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # ラベル描画
        label = f"{class_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return image


def main():
    parser = argparse.ArgumentParser(description='YOLO26n ONNX Inference')
    parser.add_argument('--model', type=str, default='yolo26n.onnx', help='ONNXモデルパス')
    parser.add_argument('--input', type=str, default='test.jpg', help='入力画像パス')
    parser.add_argument('--output', type=str, default='result.jpg', help='出力画像パス')
    parser.add_argument('--conf', type=float, default=0.25, help='信頼度閾値')
    args = parser.parse_args()

    # モデル読み込み
    print(f"モデル読み込み中: {args.model}")
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name

    # 画像読み込み
    print(f"画像読み込み中: {args.input}")
    image = cv2.imread(args.input)
    if image is None:
        raise ValueError(f"画像が読み込めません: {args.input}")
    print(f"入力画像サイズ: {image.shape[1]}x{image.shape[0]}")

    # 前処理
    blob, scale, pad_w, pad_h = preprocess(image)

    # 推論
    print("推論中...")
    outputs = session.run(None, {input_name: blob})

    # 後処理
    detections = postprocess(outputs, scale, pad_w, pad_h, conf_threshold=args.conf)
    print(f"検出数: {len(detections)}")

    # 検出結果を表示
    for det in detections:
        class_id = det['class_id']
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        print(f"  {class_name}: {det['score']:.2f} @ {det['bbox']}")

    # 結果を描画・保存
    result_image = draw_detections(image.copy(), detections)
    cv2.imwrite(args.output, result_image)
    print(f"\n結果を保存しました: {args.output}")


if __name__ == '__main__':
    main()
