from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np
import torch
import random
import cv2


def glob_file_list(root):
    """列出目录下所有.npy或.png文件"""
    return sorted(glob.glob(os.path.join(root, '*.npy')) + glob.glob(os.path.join(root, '*.png')))


def flip(x, dim):
    """沿指定维度翻转张量"""
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def augment_torch(img_list, hflip=True, rot=True):
    """随机水平翻转和旋转图像"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = flip(img, 2)
        if vflip:
            img = flip(img, 1)
        return img

    return [_augment(img) for img in img_list]


class ImagesDataset2(Dataset):
    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, train=1):
        """
        初始化数据集（直接加载指定路径下的所有子文件夹数据，不按名称筛选）
        Args:
            source_root: 源图像(低质量)根目录（训练/测试路径已通过命令行区分）
            target_root: 目标图像(高质量)根目录
            opts: 配置选项
            target_transform: 目标图像变换
            source_transform: 源图像变换
            train: 是否为训练集（仅用于数据增强开关，不影响数据筛选）
        """
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.train = train

        # 直接使用传入的路径（训练集/测试集已通过命令行参数区分）
        self.source_root = source_root
        self.target_root = target_root

        # 检查路径是否存在
        if not os.path.exists(self.source_root):
            raise FileNotFoundError(f"源图像路径不存在: {self.source_root}")
        if not os.path.exists(self.target_root):
            raise FileNotFoundError(f"目标图像路径不存在: {self.target_root}")

        # 收集所有源图像和目标图像路径
        self.source_paths = []
        self.target_paths = []

        # 列出源和目标目录下的所有子文件夹（不筛选，全部加载）
        subfolders_LQ = sorted([
            f for f in os.listdir(self.source_root)
            if os.path.isdir(os.path.join(self.source_root, f))
        ])
        subfolders_GT = sorted([
            f for f in os.listdir(self.target_root)
            if os.path.isdir(os.path.join(self.target_root, f))
        ])

        # 确保子文件夹数量匹配（源和目标目录的子文件夹一一对应）
        if len(subfolders_LQ) != len(subfolders_GT):
            raise ValueError(
                f"源目录子文件夹数量({len(subfolders_LQ)})与目标目录({len(subfolders_GT)})不匹配!"
            )

        # 遍历所有子文件夹，收集图像对（不再按名称筛选，全部加载）
        for lq_folder, gt_folder in zip(subfolders_LQ, subfolders_GT):
            lq_folder_path = os.path.join(self.source_root, lq_folder)
            gt_folder_path = os.path.join(self.target_root, gt_folder)

            # 列出子文件夹中的所有图像（.npy或.png）
            lq_images = glob_file_list(lq_folder_path)
            gt_images = glob_file_list(gt_folder_path)

            # 跳过空文件夹
            if len(lq_images) == 0 or len(gt_images) == 0:
                print(f"警告：子文件夹 {lq_folder} 或 {gt_folder} 为空，已跳过")
                continue

            # 每个低质量图像对应目标文件夹中的第一个高质量图像（可根据实际数据调整）
            for lq_img in lq_images:
                self.source_paths.append(lq_img)
                self.target_paths.append(gt_images[0])

        # 验证数据集不为空
        if len(self.source_paths) == 0:
            raise ValueError(f"未找到有效图像对！请检查路径: {self.source_root} 和 {self.target_root}")

        print(f"成功加载 {len(self.source_paths)} 对图像（{self.source_root} → {self.target_root}）")

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        """获取一个训练样本"""
        # 读取源图像(低质量)
        from_path = self.source_paths[index]
        if from_path.endswith('.npy'):
            from_im = np.load(from_path)
            from_im = from_im[:, :, [2, 1, 0]]  # BGR→RGB
            from_im = Image.fromarray(from_im)
        else:  # PNG格式
            from_im = Image.open(from_path).convert('RGB')

        # 读取目标图像(高质量)
        to_path = self.target_paths[index]
        if to_path.endswith('.npy'):
            to_im = np.load(to_path)
        else:  # PNG格式
            to_im = np.array(Image.open(to_path).convert('RGB'))

        # 生成边缘图（辅助训练）
        to_im_gray = cv2.cvtColor(to_im, cv2.COLOR_BGR2GRAY)
        sketch = cv2.GaussianBlur(to_im_gray, (3, 3), 0)
        v = np.median(sketch)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        sketch = cv2.Canny(sketch, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sketch = cv2.dilate(sketch, kernel)
        sketch = np.expand_dims(sketch, axis=-1)
        sketch = np.concatenate([sketch, sketch, sketch], axis=-1)
        assert len(np.unique(sketch)) == 2, "边缘图应仅包含0和255"

        # 转换目标图像为PIL格式并调整通道
        to_im = to_im[:, :, [2, 1, 0]]  # BGR→RGB
        to_im = Image.fromarray(to_im)

        # 应用变换
        if self.target_transform:
            to_im = self.target_transform(to_im)
        if self.source_transform:
            from_im = self.source_transform(from_im)

        # 训练时数据增强（仅训练集启用）
        if self.train and random.randint(0, 1):
            to_im = flip(to_im, 2)
            from_im = flip(from_im, 2)
            sketch = cv2.flip(sketch, 1)

        # 归一化到[0,1]
        to_im = (to_im + 1) * 0.5
        from_im = (from_im + 1) * 0.5

        # 调整边缘图尺寸并转换为张量
        height, width = to_im.shape[1], to_im.shape[2]
        sketch[sketch == 255] = 1
        sketch = cv2.resize(sketch, (width, height))
        sketch = torch.from_numpy(sketch).permute(2, 0, 1)
        sketch = sketch[0:1, :, :].long()

        return from_im, to_im, sketch