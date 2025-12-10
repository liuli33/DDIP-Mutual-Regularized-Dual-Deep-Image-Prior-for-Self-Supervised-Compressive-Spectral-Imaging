import matplotlib
matplotlib.use('Agg')  # 必须在导入 plt 之前
import matplotlib.pyplot as plt
import torch
import numpy as np
# 设置支持中文的字体（以Windows系统为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_input(tensize, const=10.0, device="cuda"):
    inp = torch.rand(tensize)/const
    inp = torch.autograd.Variable(inp, requires_grad=False).to(device)
    inp = torch.nn.Parameter(inp, requires_grad=False)
    return inp

def get_fractal_input(tensize, const=10.0, octaves=6, persistence=0.2,
                      lacunarity=2.0, base_scale=0.5, device="cuda"):
    """
    修正后的分形噪声生成函数（维度匹配版）
    """
    tensize = tuple(tensize)
    noise = torch.zeros(tensize, device=device)
    amplitude = 1.0

    # 显式提取批次和通道维度
    batch = tensize[0] if len(tensize) >= 4 else 1
    channels = tensize[1] if len(tensize) >= 4 else tensize[0]

    for octave in range(octaves):
        current_scale = base_scale * (lacunarity ** octave)
        h = max(2, int(tensize[-2] * current_scale))
        w = max(2, int(tensize[-1] * current_scale))

        # 构造4D基础噪声 [batch, channels, H, W]
        base_dims = (batch, channels, h, w)
        base = torch.randn(base_dims, device=device) * amplitude

        # 插值操作（输入为4D）
        layer = torch.nn.functional.interpolate(
            base,
            size=(tensize[-2], tensize[-1]),
            mode='bilinear',
            align_corners=False
        )

        noise += layer
        amplitude *= persistence

    noise = noise / (1.0 - (persistence ** octaves))
    return torch.nn.Parameter(noise / const, requires_grad=False)

def visualize_fractal_noise(tensize=[1, 31, 512, 512], **kwargs):
    """
    分形噪声可视化工具（支持空间域和频域分析）

    参数示例：
    tensize = [1, 3, 256, 256]  # 生成RGB噪声
    kwargs = {'octaves':6, 'persistence':0.5}
    """
    # 生成分形噪声
    noise = get_fractal_input(tensize, **kwargs)
    # noise = get_input(tensize)
    # 转换为适合可视化的格式
    if len(tensize) == 4:
        img = noise[0].permute(1, 2, 0).cpu().detach().numpy()  # [H,W,C]
    else:
        img = noise.squeeze().cpu().detach().numpy()  # [H,W]

    # 频域分析
    fft = torch.fft.fftshift(torch.fft.fft2(noise))
    spectrum = torch.log(torch.abs(fft) + 1e-9).squeeze().cpu().numpy()

    # 可视化
    plt.figure(figsize=(15, 5))

    # 空间域
    plt.subplot(131)
    if img.shape[-1] > 3:  # 多光谱取平均
        plt.imshow(img.mean(axis=-1), cmap='viridis')
    else:
        plt.imshow(img)
    # plt.title('空间域')
    plt.axis('off')

    # 频域
    plt.subplot(132)
    plt.imshow(spectrum, cmap='jet')
    # plt.title('频域能量分布')
    plt.axis('off')

    # 三维表面图（仅灰度噪声）
    if len(img.shape) == 2:
        ax = plt.subplot(133, projection='3d')
        x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        ax.plot_surface(x, y, img, cmap='terrain')
        # ax.set_title('三维表面')

    plt.savefig('分形噪声可视化.png',
                bbox_inches='tight',
                dpi=300,
                facecolor='white')


# 使用示例
visualize_fractal_noise(
    tensize=[1, 1, 256, 256],  # 生成单通道噪声
    octaves=6,
    persistence=0.01,
    lacunarity=2.0,
    base_scale=0.5,
    const=10.0
)