import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from Alexnet import Alexnet

def process_frame(frame, transformations):
    """Process a video frame for model prediction."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    image = Image.fromarray(frame)
    image = transformations(image).unsqueeze(0)  # 应用转换并添加批次维度
    return image.cuda()  # CUDA

def main():
    # load model
    MODEL_PATH = "../checkpoints/tsr_alexnet_30epochs.pth"
    num_classes = 43
    model = Alexnet(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.cuda()
    model.eval()

    transformations = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor()
    ])

    # 定义标签映射并读取真实标签
    labels = [int(x) for x in sorted([str(i) for i in range(num_classes)])]

    # class name 映射
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
    # Video capture
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # resize窗口大小
        desired_width = 960
        desired_height = 720
        frame = cv2.resize(frame, (desired_width, desired_height))

        # 预测
        image = process_frame(frame, transformations)
        with torch.no_grad():
            y_test_pred = model(image)
            y_pred_softmax = torch.log_softmax(y_test_pred[0], dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            pred_class = labels[y_pred_tags.cpu().numpy()[0]]

        # 显示预测
        pred_class_name = classes.get(pred_class, "Unknown")
        cv2.putText(frame, f'Predicted Class: {pred_class_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
