import torch
import torch.nn as nn
import onnx

# AlexNet
class Alexnet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*7*7, 1000),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x

# 43 classes
numClasses = 43

# 加载模型
model = Alexnet(numClasses)
model.load_state_dict(torch.load('../checkpoints/tsr_alexnet_30epochs.pth'))

# eval mode
model.eval() 

# 输入tensor
example_input = torch.randn(1, 3, 112, 112)

# output onnx
output_onnx_file = '../checkpoints/tsr_alexnet_30epochs_v2.onnx'
torch.onnx.export(model,
                  example_input,
                  output_onnx_file,
                  export_params=True,
                  opset_version=13,  # 可选
                  do_constant_folding=True,  # 优化模型
                  input_names=['input'],
                  output_names=['output'])

# 验证模型
onnx_model = onnx.load(output_onnx_file)
onnx.checker.check_model(onnx_model)

print("ONNX output successful!")
