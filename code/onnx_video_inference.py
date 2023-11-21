import cv2
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def process_frame(frame, transformations):
    """Process a video frame for model prediction."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 2 RGB
    image = Image.fromarray(frame)
    image = transformations(image).unsqueeze(0)
    return image

def main():
    # load onnx model
    MODEL_PATH = "../checkpoints/tsr_alexnet_30epochs_v3.onnx"
    onnx_session = ort.InferenceSession(MODEL_PATH)

    num_classes = 43
    transformations = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 真实tag映射
    labels = [int(x) for x in sorted([str(i) for i in range(num_classes)])]

    # label与meaning映射
    classes = {
        0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
        3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
        6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
        9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection', 
        12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
        16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution', 
        19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
        22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
        25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 
        28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow', 
        31: 'Wild animals crossing', 32: 'End speed + passing limits', 
        33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only', 
        36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right', 
        39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 
        42: 'End no passing veh > 3.5 tons'
    }

    # 调用web_cam
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # resize (..可选)
        desired_width = 960
        desired_height = 720
        frame = cv2.resize(frame, (desired_width, desired_height))

        # 预测
        image = process_frame(frame, transformations)
        inputs = {onnx_session.get_inputs()[0].name: np.array(image, dtype=np.float32)}
        preds = onnx_session.run(None, inputs)
        pred_class = labels[np.argmax(preds, axis=2)[0][0]]

        # show pred results
        pred_class_name = classes.get(pred_class, "Unknown")
        cv2.putText(frame, f'Predicted Class: {pred_class_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
