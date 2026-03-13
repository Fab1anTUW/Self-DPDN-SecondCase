import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# ================= CONFIG =================

DATASET_ROOT = "test_data"  # root with multiple scenes

OUTPUT_CAMERA = "data/camera/train"
OUTPUT_MODELS = "data/obj_models/train"

#create an empty val folder, enabling data_processing.py
VAL_DIR = "data/camera/val"

CLASS_ID = 1
FRAMES_PER_SCENE = 10
MASK_PARTS = 10

NUM_WORKERS = 8  #according to CPU cores

# ================= HELPERS =================

def safe_imread(path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, flags)
    if img is None or img.size == 0:
        return None
    return img

def load_instance_masks(base, source_mask):
    masks = {}
    has_background = False

    for inst in range(MASK_PARTS):
        path = os.path.join(source_mask, f"{base}_{inst:06d}.png")
        m = safe_imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        if np.any(m > 0):
            masks[inst] = m

        if np.any(m == 255):
            has_background = True

    return masks, has_background

def combine_instance_masks(masks):
    if not masks:
        return None

    h, w = next(iter(masks.values())).shape
    combined = np.full((h, w), 255, dtype=np.uint8)

    for inst, m in masks.items():
        combined[m > 0] = inst

    if not np.any(combined == 255):
        combined[[0, -1], :] = 255
        combined[:, [0, -1]] = 255

    return combined

def compute_bboxes(masks, coord):
    bboxes = {}
    for inst, mask in masks.items():
        ys, xs = np.where(mask > 0)
        coords_sel = coord[ys, xs]
        valid = np.any(coords_sel > 0, axis=1)
        coords_sel = coords_sel[valid]
        if coords_sel.size == 0:
            continue
        bboxes[inst] = np.vstack([
            coords_sel.max(axis=0),
            coords_sel.min(axis=0)
        ])
    return bboxes

# ================= FRAME WORKER =================

def process_frame(args):
    (
        dataset_path,
        base,
        global_scene_id,
        frame_id,
        skip_counters
    ) = args

    source_rgb   = os.path.join(dataset_path, "rgb")
    source_depth = os.path.join(dataset_path, "depth")
    source_mask  = os.path.join(dataset_path, "mask")
    source_nocs  = os.path.join(dataset_path, "nocs")

    # ---------- SAFETY CHECKS ----------
    rgb_img = safe_imread(os.path.join(source_rgb, base + ".png"), cv2.IMREAD_COLOR)
    if rgb_img is None:
        skip_counters["rgb_unreadable"] += 1
        return
    
    depth_img = safe_imread(os.path.join(source_depth, base + ".png"), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        skip_counters["depth_unreadable"] += 1
        return

    coord_img = safe_imread(os.path.join(source_nocs, base + ".png"), cv2.IMREAD_COLOR)
    if coord_img is None:
        skip_counters["nocs_unreadable"] += 1
        return

    # ---------- LOAD MASKS ----------
    masks, has_background = load_instance_masks(base, source_mask)
    if not masks:
        skip_counters["no_masks"] += 1
        return

    if not has_background:
        skip_counters["no_background"] += 1
        return

    # ---------- BBOX ----------
    coord = coord_img[:, :, ::-1].astype(np.float32) / 255.0
    bboxes = compute_bboxes(masks, coord)

    # ---------- WRITE MODEL BBOXES ----------
    for inst_id, bbox in bboxes.items():
        model_name = f"model_{inst_id:06d}"
        out_dir = os.path.join(
            OUTPUT_MODELS,
            f"{global_scene_id:05d}",
            model_name
        )
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, "bbox.txt"), bbox, fmt="%.6f")

    # ---------- CAMERA OUTPUT ----------
    scene_dir = os.path.join(OUTPUT_CAMERA, f"{global_scene_id:05d}")
    os.makedirs(scene_dir, exist_ok=True)

    base_out = os.path.join(scene_dir, f"{frame_id:04d}")

    # --- meta ---
    with open(base_out + "_meta.txt", "w") as f:
        for inst_id in sorted(bboxes.keys()):
            model_name = f"model_{inst_id:06d}"
            f.write(f"{inst_id} {CLASS_ID} {global_scene_id:05d} {model_name}\n")

    #resize to NOCS dataset format
    TARGET_W = 640
    TARGET_H = 480

    # --- write images (resized) ---
    rgb_resized = cv2.resize(rgb_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    coord_resized = cv2.resize(coord_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

    # --- write images ---
    cv2.imwrite(base_out + "_color.png", rgb_resized)
    cv2.imwrite(base_out + "_depth.png", depth_resized)
    cv2.imwrite(base_out + "_coord.png", coord_resized)

    

    # --- write mask ---
    combined_mask = combine_instance_masks(masks)
    mask_resized = cv2.resize(combined_mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(base_out + "_mask.png", mask_resized)

# ================= MAIN =================

def main():
    os.makedirs(OUTPUT_CAMERA, exist_ok=True)
    os.makedirs(OUTPUT_MODELS, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    manager = Manager()
    skip_counters = manager.dict({
        "rgb_unreadable": 0,
        "depth_unreadable": 0,
        "nocs_unreadable": 0,
        "no_masks": 0,
        "no_background": 0
    })

    dataset_dirs = sorted(
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    )

    global_scene_id = 0

    print(f"Found {len(dataset_dirs)} datasets")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []

        for dataset in dataset_dirs:
            dataset_path = os.path.join(DATASET_ROOT, dataset)
            rgb_dir = os.path.join(dataset_path, "rgb")
            rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))

            for idx, file in enumerate(rgb_files):
                base = os.path.splitext(file)[0]
                frame_id = idx % FRAMES_PER_SCENE

                if idx > 0 and frame_id == 0:
                    global_scene_id += 1

                futures.append(
                    executor.submit(
                        process_frame,
                        (
                            dataset_path,
                            base,
                            global_scene_id,
                            frame_id,
                            skip_counters
                        )
                    )
                )

            global_scene_id += 1

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    # ---------- Skip summary ----------
    print("\n=== SKIP SUMMARY ===")
    for k, v in skip_counters.items():
        print(f"{k}: {v}")

    print("\n=== ALL DATASETS CONVERTED ===")


if __name__ == "__main__":
    main()
