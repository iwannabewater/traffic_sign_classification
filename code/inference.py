import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import pandas as pd
from Alexnet import Alexnet

def load_image(image_path):
    """Load an image, convert it to RGB, and apply necessary transformations."""
    transformations = transforms.Compose([
        transforms.Resize([112, 112]),  # Resize the image
        transforms.ToTensor()           
    ])
    image = Image.open(image_path).convert('RGB')  # to RGB
    image = transformations(image).unsqueeze(0)
    return image

def predict(model, image, labels):
    """Make a prediction for a single image."""
    model.eval()
    with torch.no_grad():
        image = image.cuda()
        y_test_pred = model(image)

        # 应用 log_softmax 并找到预测标签
        y_pred_softmax = torch.log_softmax(y_test_pred[0], dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        y_pred_tags = y_pred_tags.cpu().numpy()

        y_pred = y_pred_tags[0]
        y_pred = labels[y_pred]  # 映射原始标签
        return y_pred

def main(image_paths):
    # 加载模型
    MODEL_PATH = "../checkpoints/tsr_alexnet_30epochs.pth"
    num_classes = 43  # 43 classes
    model = Alexnet(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model = model.cuda()
    # model = model.to('cpu')

    # 定义标签映射并读取真实标签
    labels = [int(x) for x in sorted([str(i) for i in range(num_classes)])]
    labels_csv = "../csv_files/test.csv"  # 读取csv文件
    df = pd.read_csv(labels_csv)
    gt_labels = dict(zip(df['Path'].apply(lambda x: x.split('/')[-1]), df['ClassId']))

    # 定义序号与标签映射
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

    # 预测图像
    for image_path in image_paths:
        image = load_image(image_path)
        pred_class = predict(model, image, labels)
        image_file_name = image_path.split('/')[-1]
        gt_class = gt_labels.get(image_file_name, None)

        # 打印预测
        pred_class_name = classes.get(pred_class, "Unknown")
        gt_class_name = classes.get(gt_class, "Unknown")

        print(f"Prediction for {image_path}: Predicted Class: {pred_class_name}, Ground Truth Class: {gt_class_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument('images', type=str, nargs='+', help='Image file paths')
    args = parser.parse_args()
    main(args.images)
