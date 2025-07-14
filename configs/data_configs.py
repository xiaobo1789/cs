from configs import transforms_config
from configs.paths_config import dataset_paths
from utils.data_transforms import DataTransforms

DATASETS = {
	'ours_encode': {
		'train_source_root': 'E:/SMG-LLIE-main/datasets/SID/SMID_LQ_np',  # 训练低光照路径（需存在）
        'train_target_root': 'E:/SMG-LLIE-main/datasets/SID/SMID_Long_np',  # 训练正常光照路径（需存在）
        'test_source_root': 'E:/SMG-LLIE-main/datasets/SID/SMID_LQ_test_np',  # 测试低光照路径（需存在）
        'test_target_root': 'E:/SMG-LLIE-main/datasets/SID/SMID_Long_test_np',
	},
}

def get_dataset_config(opts):
    return {
        'ours_encode': {
            'transforms': transforms_config.OursEncodeTransforms,
            'source_root': opts.source_root,
            'target_root': opts.target_root,
            'test_source_root': opts.test_source_root,
            'test_target_root': opts.test_target_root,
        },
    }