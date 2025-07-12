dataset_paths = {
	'ours_train_lq': r"E:\SMG-LLIE-main\datasets\train\low",  # 替换为你的低光照训练集绝对路径
    'ours_train_hq': r"E:\SMG-LLIE-main\datasets\train\high", # 替换为你的正常光照训练集绝对路径
    # 测试集：低光照和正常光照图片路径
    'ours_test_lq': r"E:\SMG-LLIE-main\datasets\test\low",    # 替换为你的低光照测试集绝对路径
    'ours_test_hq': r"E:\SMG-LLIE-main\datasets\test\high",   # 替换为你的正常光照测试集绝对路径
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt'
}

