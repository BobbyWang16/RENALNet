import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        if isinstance(kernel_size, tuple):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual  # 非in-place操作


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=False),
            nn.Linear(reduced_channels, channels)
        )
        
    def forward(self, x):
        b, c = x.size(0), x.size(1)
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        return torch.sigmoid(out).view(b, c, 1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, kernel_size), 
                             padding=(padding, padding, padding))
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))


class MaskAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x, mask=None):
        # 通道注意力
        channel_att = self.channel_att(x)
        x = torch.mul(x, channel_att)
        
        # 空间注意力
        spatial_att = self.spatial_att(x)
        x = torch.mul(x, spatial_att)
        
        # 应用mask
        if mask is not None:
            if len(mask.shape) == 4:
                mask = mask.unsqueeze(1)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(0).unsqueeze(0)
                
            target_size = x.shape[2:]
            mask = mask.to(x.dtype)
            
            if mask.shape[-3] == 1:
                mask = mask.repeat(1, 1, target_size[0], 1, 1)
            
            mask = F.interpolate(mask, size=target_size, mode='nearest')
            x = torch.mul(x, mask)
            
        return x


class MaskAttention3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        
        # 初始特征提取
        self.input_conv = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=(3, 7, 7), stride=(1, 2, 2)),
            # nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # 主干网络
        self.layer1 = nn.ModuleList([
            ConvBlock(16, 32, stride=2),
            ResBlock(32),
            MaskAttentionModule(32)
        ])
        
        self.layer2 = nn.ModuleList([
            ConvBlock(32, 64, stride=2),
            ResBlock(64),
            MaskAttentionModule(64)
        ])
        
        self.layer3 = nn.ModuleList([
            ConvBlock(64, 128, stride=2),
            ResBlock(128),
            MaskAttentionModule(128)
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
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
    
    def forward(self, x, mask=None):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.float32)
        # print(x.shape)
        x = self.input_conv(x)
        # print(x.shape)
        
        # 主干网络处理
        for layer in [self.layer1, self.layer2, self.layer3]:
            x = layer[0](x)  # ConvBlock
            x = layer[1](x)  # ResBlock
            x = layer[2](x, mask)  # MaskAttentionModule
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


import torch
import torch.nn as nn
from torchviz import make_dot
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def visualize_model():
    # 1. 创建模型实例
    model = MaskAttention3DCNN()
    
    # 2. 生成示例输入
    batch_size = 2
    channels = 1
    depth = 16
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, depth, height, width)
    
    # 3. 模型结构可视化
    def visualize_structure():
        # 使用torchviz生成计算图
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render("ma_3dcnn_structure", format="pdf")
        
        # 打印模型结构
        print("Model Structure:")
        print(model)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    # 4. 特征图可视化
    def visualize_feature_maps():
        # 注册钩子来获取中间特征图
        feature_maps = {}
        
        def hook_fn(module, input, output):
            feature_maps[module] = output.detach()
        
        # 为每个主要层注册钩子
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                module.register_forward_hook(hook_fn)
        
        # 前向传播
        with torch.no_grad():
            model(x)
        
        # 可视化特征图
        for layer, feature_map in feature_maps.items():
            # 选择第一个样本的特征图
            fm = feature_map[0]  # (C, D, H, W)
            
            # 创建网格图
            num_channels = min(16, fm.size(0))  # 最多显示16个通道
            fig, axes = plt.subplots(4, 4, figsize=(15, 15))
            axes = axes.ravel()
            
            for idx in range(num_channels):
                # 选择中间的深度切片
                middle_slice = fm[idx, fm.size(1)//2, :, :]
                axes[idx].imshow(middle_slice.cpu().numpy(), cmap='viridis')
                axes[idx].axis('off')
                axes[idx].set_title(f'Channel {idx}')
            
            plt.savefig(f'feature_maps_{layer}.png')
            plt.close()
    
    # 5. 注意力图可视化
    def visualize_attention():
        # 获取注意力权重
        attention_maps = {}
        
        def attention_hook(module, input, output):
            if isinstance(module, nn.Sigmoid):  # 假设注意力图经过Sigmoid激活
                attention_maps[module] = output.detach()
        
        # 注册钩子
        for name, module in model.named_modules():
            if "attention" in name.lower():
                module.register_forward_hook(attention_hook)
        
        # 前向传播
        with torch.no_grad():
            model(x)
        
        # 可视化注意力图
        for layer, attention_map in attention_maps.items():
            # 选择第一个样本
            att_map = attention_map[0]  # (C, D, H, W) or (1, D, H, W)
            
            if len(att_map.shape) == 4:
                # 空间注意力图
                middle_slice = att_map[0, att_map.size(1)//2, :, :]
                plt.figure(figsize=(8, 8))
                plt.imshow(middle_slice.cpu().numpy(), cmap='hot')
                plt.colorbar()
                plt.title(f'Attention Map - {layer}')
                plt.savefig(f'attention_map_{layer}.png')
                plt.close()
    
    # 6. TensorBoard可视化
    def visualize_tensorboard():
        writer = SummaryWriter('runs/ma_3dcnn')
        
        # 添加模型图
        writer.add_graph(model, x)
        
        # 添加参数分布
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
        
        writer.close()
    
    # 7. 权重分布可视化
    def visualize_weights():
        plt.figure(figsize=(15, 5))
        
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'weight' in name and 'conv' in name:
                weights = param.clone().cpu().data.numpy().flatten()
                plt.subplot(1, 3, i%3 + 1)
                plt.hist(weights, bins=50)
                plt.title(f'Weight Distribution - {name}')
                
        plt.tight_layout()
        plt.savefig('weight_distributions.png')
        plt.close()

    # 执行所有可视化函数
    visualize_structure()
    visualize_feature_maps()
    visualize_attention()
    visualize_tensorboard()
    visualize_weights()

def visualize_training_process():
    # 模拟训练过程中的指标变化
    epochs = 100
    train_losses = np.random.rand(epochs) * 0.5 + 0.5
    train_accs = np.random.rand(epochs) * 0.2 + 0.7
    val_losses = np.random.rand(epochs) * 0.5 + 0.6
    val_accs = np.random.rand(epochs) * 0.15 + 0.75
    
    # 绘制训练过程曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.close()

# if __name__ == "__main__":
#     visualize_model()
#     # visualize_training_process()

def test_model():
    # 测试模型
    model = MaskAttention3DCNN(in_channels=1, num_classes=1)
    batch_size, channels, depth, height, width = 2, 1, 64, 224, 224
    x = torch.randn(batch_size, channels, depth, height, width)
    mask = torch.ones(batch_size, height, width)
    output = model(x, mask)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    return model


if __name__ == "__main__":
    model = test_model()