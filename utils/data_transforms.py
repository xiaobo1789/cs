import torchvision.transforms as transforms

class DataTransforms:
    def __init__(self, opts):
        self.opts = opts
        self.image_size = opts.stylegan_size  # 从参数获取图像尺寸

    def get_transforms(self):
        # 定义训练/测试的图像变换（Resize→ToTensor→归一化）
        common_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return {
            'transform_source': common_transform,  # 低光照图像变换
            'transform_gt_train': common_transform,  # 正常光照训练图像变换
            'transform_test': common_transform  # 测试图像变换
        }