from configs import transforms_config
from configs.paths_config import dataset_paths
from utils.data_transforms import DataTransforms

DATASETS = {
	'ours_encode': {
		'transforms': transforms_config.OursEncodeTransforms,
		'train_source_root': dataset_paths['source_root'],
		'train_target_root': dataset_paths['target_root'],
		'test_source_root': dataset_paths['test_source_root'],
		'test_target_root': dataset_paths['test_target_root'],
	},
}

def get_dataset_config(opts):
    return {
        'ours_encode': {
            'transforms': transforms_config.OursEncodeTransforms,
            'train_source_root': opts.source_root,
            'train_target_root': opts.target_root,
            'test_source_root': opts.test_source_root,
            'test_target_root': opts.test_target_root,
        },
    }