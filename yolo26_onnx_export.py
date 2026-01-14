from ultralytics import YOLO

# モデルをロード
model = YOLO("yolo26n.pt")
# ONNXにエクスポート
model.export(format="onnx", opset=17)

