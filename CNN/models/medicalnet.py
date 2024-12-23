import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
#         super().__init__()
#         self.conv = nn.Conv3d(
#             in_channels, out_channels, kernel_size, 
#             stride=stride, padding=kernel_size//2, bias=False
#         )
#         self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
#         self.relu = nn.ReLU(inplace=True)
#         self.se = SEBlock(out_channels)
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.se(x)
#         return x
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # 如果kernel_size是tuple，则padding也需要是tuple
        if isinstance(kernel_size, tuple):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2
            
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x

class Medical3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        
        # 初始特征提取
        self.input_conv = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=(3, 7, 7), stride=(1, 2, 2)),  # 112x112x32
            # ConvBlock(in_channels, 32, kernel_size=7, stride=2),           # 112x112x16
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)      # 56x56x16
        )
        
        # 主干网络
        self.layer0 = nn.Sequential(
            ConvBlock(16, 32, stride=2),    # 28x28x8
            ResBlock(32),
        )

        # 主干网络
        self.layer1 = nn.Sequential(
            ConvBlock(32, 64, stride=2),    # 28x28x8
            ResBlock(64),
        )
        
        self.layer2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),   # 14x14x4
            ResBlock(128),
        )
        
        # self.layer3 = nn.Sequential(
        #     ConvBlock(128, 256, stride=2),  # 7x7x2
        #     ResBlock(256),
        # )
        
        # 全局特征
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            # nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入归一化
        # x = (x - x.mean()) / (x.std() + 1e-6)
        
        # 特征提取
        x = self.input_conv(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        
        # 全局池化
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x

def get_model(in_channels=1, num_classes=1):
    model = Medical3DCNN(in_channels=in_channels, num_classes=num_classes)
    return model

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = get_model()
    
    # 测试数据
    batch_size = 2
    x = torch.randn(batch_size, 1, 32, 224, 224)
    
    # 前向传播
    output = model(x)
    # print(output)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")