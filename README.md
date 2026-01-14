# YOLO26n ONNX 推論

YOLO26n ONNXモデルを使用した物体検出プロジェクト。
<img width="400" height="266" alt="image" src="https://github.com/user-attachments/assets/f0389e60-d84c-4c45-9f42-605aaea5a562" />

### ONNXモデル
https://netron.app/?url=https://raw.githubusercontent.com/nnn112358/yolo26_onnxruntime_example/main/yolo26n.onnx


## モデル情報

| 項目 | 値 |
|------|-----|
| モデル名 | yolo26n.onnx |
| ファイルサイズ | 9.48 MB |
| パラメータ数 | 2,450,977 |
| 入力サイズ | 640x640 |
| 入力形状 | [1, 3, 640, 640] (NCHW) |
| 出力形状 | [1, 300, 6] |
| Opset Version | 17 |
| Producer | PyTorch 2.9.1 |

### 出力フォーマット

出力は最大300件の検出結果を含み、各検出は以下の6値で構成:
- `x1, y1, x2, y2`: バウンディングボックス座標
- `score`: 信頼度スコア
- `class_id`: COCOクラスID (0-79)

## セットアップ

```bash
# 依存関係のインストール
uv sync
```

## 使用方法

### 基本的な使い方

```bash
uv run python inference.py
```

### オプション指定

```bash
uv run python inference.py --input 入力画像.jpg --output 出力画像.jpg --conf 0.5
```

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--model` | yolo26n.onnx | ONNXモデルパス |
| `--input` | test.jpg | 入力画像パス |
| `--output` | result.jpg | 出力画像パス |
| `--conf` | 0.25 | 信頼度閾値 |

## 依存関係

- Python 3.12+
- onnxruntime
- opencv-python
- numpy

## ファイル構成

```
.
├── README.md           # このファイル
├── inference.py        # 推論スクリプト
├── yolo26n.onnx        # ONNXモデル
├── yolo26n.pt          # PyTorchモデル (元モデル)
├── yolo26_onnx_export.py  # ONNXエクスポートスクリプト
├── test.jpg            # テスト画像
├── result.jpg          # 推論結果
├── pyproject.toml      # プロジェクト設定
└── uv.lock             # 依存関係ロックファイル
```

## 検出可能なクラス

COCOデータセットの80クラスを検出可能:

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
