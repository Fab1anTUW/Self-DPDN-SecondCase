# Outdated Version for online 1 inference file at a time
# use make_results_pkl_iterate.py

import os
import cv2
import glob
import pickle
import numpy as np

DATA_ROOT = "data"
CAMERA_ROOT = os.path.join(DATA_ROOT, "camera", "val")
OBJ_ROOT = os.path.join(DATA_ROOT, "obj_models/train")
OUT_ROOT = os.path.join(DATA_ROOT, "detection", "CAMERA25")
PKL_PATH = "data/camera/val/00000/0000_label.pkl"

os.makedirs(OUT_ROOT, exist_ok=True)

pkl_idx = 0

scene_dirs = sorted(glob.glob(os.path.join(CAMERA_ROOT, "*")))

for scene_dir in scene_dirs:
    frame_bases = sorted(glob.glob(os.path.join(scene_dir, "*_color.png")))

    for color_path in frame_bases:
        base = color_path.replace("_color.png", "")

        mask_path = base + "_mask.png"
        meta_path = base + "_meta.txt"

        if not os.path.exists(mask_path):
            continue

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        H, W = mask_img.shape

        instance_ids = sorted([i for i in np.unique(mask_img) if i != 255])
        if len(instance_ids) == 0:
            continue

        pred_masks = []
        pred_bboxes = []
        pred_class_ids = []
        pred_scores = []

        obj_dirs = sorted(glob.glob(os.path.join(OBJ_ROOT, "*/*/")))

        for obj_dir in obj_dirs:
            bbox_files = glob.glob(os.path.join(obj_dir, "bbox.txt"))
            if not bbox_files:
                continue

            pred_bboxes = []
            pred_class_ids = []
            pred_scores = []

            for bbox_file in bbox_files:
                with open(bbox_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:

                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x1, y1, x2, y2 = map(float, parts[:4])
                            pred_bboxes.append([x1, y1, x2, y2])
                            pred_class_ids.append(1)
                            pred_scores.append(1.0)

            if not pred_bboxes:
                continue

        for inst_id in instance_ids:
            inst_mask = (mask_img == inst_id)
            if inst_mask.sum() < 20:
                continue

            ys, xs = np.where(inst_mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            pred_masks.append(inst_mask)
            pred_bboxes.append([x1, y1, x2, y2])
            pred_class_ids.append(1)  # luggage → class 1
            pred_scores.append(1.0)

        
        mask_folder = "test_data/00000/mask"
        mask_paths = sorted(glob.glob(os.path.join(mask_folder, '*.png')))

        for mask_file in mask_paths:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask > 0).astype(np.uint8)
            if np.sum(mask_bin) < 16:
                continue
            pred_masks.append(mask_bin)

        if len(pred_masks) > 0:
            pred_masks = np.stack(pred_masks, axis=-1)  # H x W x n
        else:
            pred_masks = np.zeros((480, 640, 0), dtype=np.uint8)

        rel = os.path.relpath(base, DATA_ROOT)

        # load gt from img_pkl file

        with open(PKL_PATH, "rb") as f:
            data = pickle.load(f)
        
        rotations = data["rotations"]
        translations = data["translations"]
        sizes = data["sizes"]
        scales = data["scales"]

        num_instances = len(rotations)
        gt_RTs = np.zeros((num_instances, 4, 4), dtype=np.float32)

        for i in range(num_instances):

            R = rotations[i]
            T = translations[i]
            s = scales[i]

            RT = np.eye(4, dtype=np.float32)
            RT[:3, :3] = R * s
            RT[:3, 3] = T

            gt_RTs[i] = RT

        result = {
            'image_path' : os.path.join("NOCS", rel),
            'image_id': 1,
            'gt_class_ids': np.array(pred_class_ids, dtype=np.int32),
            'gt_bboxes': np.array(pred_bboxes, dtype=np.float32),
            'gt_RTs': gt_RTs,
            'gt_scales': sizes.astype(np.float32),
            'gt_handle_visibility': np.zeros(len(pred_class_ids), dtype=np.float32),
            'obj_list': [],

            'pred_masks': pred_masks,
            'pred_bboxes': np.array(pred_bboxes, dtype=np.float32),
            'pred_class_ids': np.array(pred_class_ids, dtype=np.int32),
            'pred_scores': np.array(pred_scores, dtype=np.float32),
            'pred_scales': sizes.astype(np.float32),
            'pred_RTs': gt_RTs
        }

        out_path = os.path.join(OUT_ROOT, f"results_{pkl_idx:06d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(result, f)

        pkl_idx += 1

print(f"[DONE] wrote {pkl_idx} result pkls")
