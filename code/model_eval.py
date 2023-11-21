import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')


test_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
    ])


test_data_path = "../datasets/test"
test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform = test_transforms)
test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)

# 43 classes
numClasses = 43

num = range(numClasses)
labels = []
for i in num:
    labels.append(str(i))
labels = sorted(labels)
for i in num:
    labels[i] = int(labels[i])
print("List of labels : ")
print("Actual labels \t--> Class in PyTorch")
for i in num:
    print("\t%d \t--> \t%d" % (labels[i], i))


# 读取csv file
df = pd.read_csv("../csv_files/test.csv")
numExamples = len(df)
labels_list = list(df.ClassId)

# 加载模型
from Alexnet import Alexnet
MODEL_PATH = "../checkpoints/tsr_alexnet_30epochs.pth"
model = Alexnet(numClasses)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()


y_pred_list = []
corr_classified = 0

with torch.no_grad():
    model.eval()

    i = 0

    for image, _ in test_loader:
        image = image.cuda()

        y_test_pred = model(image)

        y_pred_softmax = torch.log_softmax(y_test_pred[0], dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        y_pred_tags = y_pred_tags.cpu().numpy()
        
        y_pred = y_pred_tags[0]
        y_pred = labels[y_pred]
        
        y_pred_list.append(y_pred)

        if labels_list[i] == y_pred:
            corr_classified += 1

        i += 1

print("Number of correctly classified images = %d" % corr_classified)
print("Number of incorrectly classified images = %d" % (numExamples - corr_classified))
print("Final accuracy = %f" % (corr_classified / numExamples))

# classification report
print(classification_report(labels_list, y_pred_list))

# confusion matrix
def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize = (20, 20));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = classes);
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    plt.xticks(rotation = 20)
    plt.savefig("./saved_images/confusion_matrix.png", bbox_inches='tight', pad_inches=0.5)
    
labels_arr = range(0, numClasses)
plot_confusion_matrix(labels_list, y_pred_list, labels_arr)

print(y_pred_list[:20])

print(labels_list[:20])

# 选取前30张进行预测
fig, axs = plt.subplots(6, 5, figsize=(50, 75))

for i in range(30):
    row = i // 5
    col = i % 5
    
    imgName = '../datasets/test/' + df.iloc[i].Path
    img = Image.open(imgName)
    axs[row, col].imshow(img)

    pred = y_pred_list[i]
    actual = labels_list[i]
    title = "Pred: %d, Actual: %d" % (pred, actual)

    if pred == actual:
        axs[row, col].set_title(title, fontsize=50, color='green')
    else:
        axs[row, col].set_title(title, fontsize=50, color='red')

plt.savefig("./saved_images/predictions.png", bbox_inches='tight', pad_inches=0.5)
