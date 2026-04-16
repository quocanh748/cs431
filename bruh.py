import os, shutil

VAL_DIR = "/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200/val"
DEST = "/kaggle/working/val_formatted"

os.makedirs(DEST, exist_ok=True)

# Đọc file annotation
with open(os.path.join(VAL_DIR, "val_annotations.txt")) as f:
    for line in f:
        file, class_id, *_ = line.strip().split('\t')
        cls_dir = os.path.join(DEST, class_id)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(VAL_DIR, "images", file)
        dst = os.path.join(cls_dir, file)
        shutil.copy(src, dst)