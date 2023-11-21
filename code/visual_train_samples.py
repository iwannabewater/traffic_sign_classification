import os
import matplotlib.pyplot as plt
from PIL import Image
import random

numClasses = 43

fig, axs = plt.subplots(numClasses // 2 + 1, 10, figsize=(100,100))

for i in range(numClasses):
    folderName = "../datasets/train/" + str(i)
    filenames = os.listdir(folderName)
    random.shuffle(filenames)
    
    for j in range(5):
        imgName = folderName + "/" + filenames[j]
        img = Image.open(imgName)
        axs[i//2, j+(5*(i%2))].imshow(img)

plt.savefig("./saved_images/sample_train_images.png", dpi=40, bbox_inches = 'tight', pad_inches=0.5)