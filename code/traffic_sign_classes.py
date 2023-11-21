import matplotlib.pyplot as plt
from PIL import Image

import matplotlib
matplotlib.use('Agg')

numClasses = 43

fig, axs = plt.subplots(9,5, figsize=(100,150))
fig.tight_layout(h_pad = 20)
for i in range(numClasses):
    row = i // 5
    col = i % 5
    
    imgName = '../datasets/meta_file/' + str(i) + ".png"
    img = Image.open(imgName)
    axs[row, col].imshow(img)
    axs[row, col].set_title(str(i), fontsize=75)

plt.savefig("./saved_images/traffic_sign_classes.png", bbox_inches = 'tight', pad_inches=0.5)