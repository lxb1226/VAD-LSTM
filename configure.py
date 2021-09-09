import os

# 一些文件夹的配置选项
root = os.getcwd()
dev_path = os.path.join(root, "wavs", "dev")
train_path = os.path.join(root, "wavs", "train")
data_path = os.path.join(root, r"data")
fig_path = os.path.join(root, r"fig")
clean_test_path = os.path.join(root, "wavs", "test", "clean_data_test")
seen_noise_test_path = os.path.join(root, "wavs", "test", "seen_noise_test")
unseen_noise_test_path = os.path.join(root, "wavs", "test", "unseen_noise_test")
feat_path = os.path.join(root, r"feats")
dev_feat_path = os.path.join(feat_path, r'dev')
train_feat_path = os.path.join(feat_path, r'train')
# test_feat_path = os.path.join(feat_path, r'test')
clean_feat_path = os.path.join(feat_path, r'test', r'clean_data_test')
seen_noise_feat_path = os.path.join(feat_path, r'test', r'seen_noise_test')
unseen_noise_feat_path = os.path.join(feat_path, r'test', r'unseen_noise_test')

if not os.path.exists(feat_path):
    os.mkdir(feat_path)
for path in [fig_path, dev_feat_path, train_feat_path, clean_feat_path, seen_noise_feat_path,
             unseen_noise_feat_path]:
    if not os.path.exists(path):
        os.mkdir(path)
