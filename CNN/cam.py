from models.medicalnet import Medical3DCNN
from models.maskcnn import MaskAttention3DCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import RegularGridInterpolator
from util.dataset import adjust_window

def b_spline_resample_3d(data, target_size, order=3):
    """
    使用B样条插值将3D数组重采样到目标大小
    
    参数:
    data: numpy数组，形状为 [D, H, W]
    target_size: tuple，目标尺寸 (D_new, H_new, W_new)
    order: int，B样条的阶数，默认为3 (三次B样条)
    
    返回:
    resampled_data: 重采样后的numpy数组，形状为target_size
    """
    # 获取原始尺寸
    D, H, W = data.shape
    D_new, H_new, W_new = target_size
    
    # 创建原始坐标网格
    z = np.linspace(0, D-1, D)
    y = np.linspace(0, H-1, H)
    x = np.linspace(0, W-1, W)
    
    # 创建插值函数
    interpolator = RegularGridInterpolator((z, y, x), data, method='linear', 
                                         bounds_error=False, fill_value=0)
    
    # 创建目标坐标网格
    z_new = np.linspace(0, D-1, D_new)
    y_new = np.linspace(0, H-1, H_new)
    x_new = np.linspace(0, W-1, W_new)
    
    # 生成网格点
    zv, yv, xv = np.meshgrid(z_new, y_new, x_new, indexing='ij')
    
    # 进行插值
    points = np.stack([zv, yv, xv], axis=-1)
    resampled_data = interpolator(points)
    
    return resampled_data

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output.clone()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, mask_tensor, target_class=None):
        # 确保输入需要梯度
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # 前向传播
        model_output = self.model(input_tensor, mask_tensor)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
        
        # 反向传播
        self.model.zero_grad()
        score = model_output[0, target_class]
        score.backward()
        
        # 确保使用克隆的数据
        gradients = self.gradients.detach().clone()
        activations = self.activations.detach().clone()
        
        # 计算权重
        weights = torch.mean(gradients, dim=(2, 3, 4))
        
        # 生成CAM
        batch, channel = weights.shape
        cam = torch.zeros(activations.shape[2:], device=activations.device)
        
        for i in range(channel):
            cam += weights[0, i] * activations[0, i]
        
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.cpu().numpy()

def normalize_image(img):
    """归一化图像到0-1范围"""
    img = img.copy()
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def visualize_cam(image_3d, cam_3d, slice_idx):
    """为单个切片生成CAM可视化"""
    
    # image_3d = adjust_window(image_3d, 400, 40)
    # 获取对应切片
    image = image_3d[slice_idx]
    target_size = image_3d.shape
    # 将cam_3D采样到与图像相同的尺寸
    # cam_3d = cv2.resize(cam_3d, target_size)
    cam_3d = b_spline_resample_3d(cam_3d, target_size)
    cam = cam_3d[slice_idx]
    print(image_3d.shape, cam_3d.shape)
    # 归一化
    image = normalize_image(image)
    cam = normalize_image(cam)
    print(image.shape, cam.shape)
    # 调整尺寸
    # image = cv2.resize(image, target_size)
    # cam = cv2.resize(cam, target_size)
    
    # 转换为uint8
    image_uint8 = (image * 255).astype(np.uint8)
    cam_uint8 = (cam * 255).astype(np.uint8)
    
    # 生成热力图
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    
    # 将原始图像转换为RGB（确保尺寸匹配）
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    
    # 确保热力图和图像具有相同的尺寸
    assert image_rgb.shape == heatmap.shape, f"Shape mismatch: image_rgb {image_rgb.shape} vs heatmap {heatmap.shape}"
    
    # 叠加热力图
    alpha = 0.5
    overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap, alpha, 0)
    
    return overlay

def visualize_3d_cam(model, input_tensor, mask_tensor, target_class=None, num_slices=6):
    # 确保模型处于评估模式
    model.eval()
    
    # 获取目标层
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and 'layer2' in name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError("Could not find target layer")
    
    # 创建Grad-CAM实例
    gradcam = GradCAM(model, target_layer)
    
    # 生成CAM
    cam = gradcam.generate_cam(input_tensor, mask_tensor, target_class)
    print(cam.shape)
    
    # 获取原始图像
    original_image = input_tensor.squeeze().cpu().numpy()
    
    # 选择要显示的切片
    depth = original_image.shape[0]
    # slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    slice_indices = [21,23,25,27,29,31,33,35,37,39,41,43]
    
    # 创建图表
    fig, axes = plt.subplots(4, num_slices, figsize=(3*num_slices, 18))
    
    for i, slice_idx in enumerate(slice_indices):
        print(i, i//6)
        # 得余数函数

        try:
            # 显示原始图像
            if i < 6:
                axes[0, i%6].imshow(original_image[slice_idx], cmap='gray')
                axes[0, i%6].set_title(f'Original Slice {slice_idx}')
                axes[0, i%6].axis('off')
                # 显示CAM叠加结果
                overlay = visualize_cam(original_image, cam, slice_idx)
                axes[1, i%6].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[1, i%6].set_title(f'CAM Slice {slice_idx}')
                axes[1, i%6].axis('off')
            else:
                axes[2, i%6].imshow(original_image[slice_idx], cmap='gray')
                axes[2, i%6].set_title(f'Original Slice {slice_idx}')
                axes[2, i%6].axis('off')
                # 显示CAM叠加结果
                overlay = visualize_cam(original_image, cam, slice_idx)
                axes[3, i%6].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[3, i%6].set_title(f'CAM Slice {slice_idx}')
                axes[3, i%6].axis('off')
 
        except Exception as e:
            print(f"Error processing slice {slice_idx}: {str(e)}")
    
    plt.tight_layout()
    # 保存pdf
    plt.savefig('cam_3.pdf')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 创建模型并加载权重（如果有的话）
    # model = Medical3DCNN(in_channels=1, num_classes=1)
    model = MaskAttention3DCNN()
    # 如果有预训练权重，在这里加载
    model.load_state_dict(torch.load('./checkpoint/chk_1/kits_epoch90.pth'))
    
    # 确保模型处于评估模式
    model.eval()
    
    # 创建示例输入
    # input_tensor = torch.randn(1, 1, 32, 224, 224)

    input_np = np.load('./data/images/train_test/TJH0475.npy')

    input_np = adjust_window(input_np, 600, 40)
    mask_np = np.load('./data/labels/train_test/.npy')
    # mask_np[mask_np != 2] = 0
    # mask_np[mask_np == 2] = 1
    mask_np[mask_np == 3] = 1
    input_tensor = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0).float()
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
    
    try:
        visualize_3d_cam(model, input_tensor, mask_tensor, num_slices=6)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()