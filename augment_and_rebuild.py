"""
NIGRAAN - Data Augmentation and Embedding Rebuild
==================================================
Confirmed paths:
  LFW faces  : datasets/lfw/lfw-deepfunneled/
  LFW pairs  : datasets/lfw/pairs.csv
  Raw images : data/raw_images/S001_Umair_Iqbal/ etc
  Embeddings : data/embeddings/
"""

import sys
import os
import shutil
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# ── Project root setup ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from python_engine.embedding_builder import EmbeddingBuilder

# ── Confirmed paths ──────────────────────────────────────────────────────────
LFW_DIR        = os.path.join(PROJECT_ROOT, "datasets", "lfw", "lfw-deepfunneled")
AUGMENTED_DIR  = os.path.join(config.DATA_DIR, "augmented")
EMB_BACKUP_DIR = os.path.join(config.DATA_DIR, "embeddings_backup")
EMB_NEW_DIR    = os.path.join(config.DATA_DIR, "embeddings_augmented")

# ── Settings ─────────────────────────────────────────────────────────────────
AUGMENTATIONS_PER_PHOTO = 8
NUM_NEGATIVES           = 200


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BACKUP
# ─────────────────────────────────────────────────────────────────────────────
def backup_old_embeddings():
    if not os.path.exists(config.EMBEDDINGS_DIR):
        print("  No existing embeddings found, skipping backup.")
        return

    if os.path.exists(EMB_BACKUP_DIR):
        shutil.rmtree(EMB_BACKUP_DIR)

    shutil.copytree(config.EMBEDDINGS_DIR, EMB_BACKUP_DIR)
    print(f"  Backed up to: data/embeddings_backup/")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — AUGMENT
# ─────────────────────────────────────────────────────────────────────────────
def augment_single_image(img_path, output_dir, base_name, count):
    """
    Creates augmented versions of one photo.
    Simulates real conditions: lighting changes, angles, blur.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  [WARN] Could not open {img_path}: {e}")
        return 0

    # Save clean original copy into augmented folder
    img.save(os.path.join(output_dir, f"{base_name}_orig.jpg"), quality=95)

    saved = 0
    for i in range(count):
        aug = img.copy()

        # Brightness — simulates different lighting on campus
        aug = ImageEnhance.Brightness(aug).enhance(
            random.uniform(0.6, 1.4)
        )

        # Contrast
        aug = ImageEnhance.Contrast(aug).enhance(
            random.uniform(0.7, 1.3)
        )

        # Sharpness
        aug = ImageEnhance.Sharpness(aug).enhance(
            random.uniform(0.5, 1.8)
        )

        # Horizontal flip — simulates person walking in either direction
        if random.random() > 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

        # Rotation — simulates slight head tilt or camera angle
        aug = aug.rotate(
            random.uniform(-15, 15),
            resample=Image.BILINEAR,
            fillcolor=(128, 128, 128)
        )

        # Slight blur — simulates motion or camera focus variation
        if random.random() > 0.7:
            aug = aug.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(0.3, 1.2)
                )
            )

        out_path = os.path.join(
            output_dir, f"{base_name}_aug_{i:03d}.jpg"
        )
        aug.save(out_path, quality=95)
        saved += 1

    return saved


def augment_all_students():
    os.makedirs(AUGMENTED_DIR, exist_ok=True)

    if not os.path.exists(config.RAW_IMAGES_DIR):
        print(f"  [ERROR] Not found: {config.RAW_IMAGES_DIR}")
        return 0

    student_folders = [
        f for f in os.listdir(config.RAW_IMAGES_DIR)
        if os.path.isdir(os.path.join(config.RAW_IMAGES_DIR, f))
    ]

    if not student_folders:
        print(f"  [ERROR] No student folders in {config.RAW_IMAGES_DIR}")
        return 0

    total_created = 0

    for folder_name in student_folders:
        src_folder = os.path.join(config.RAW_IMAGES_DIR, folder_name)
        aug_folder = os.path.join(AUGMENTED_DIR, folder_name)
        os.makedirs(aug_folder, exist_ok=True)

        photos = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not photos:
            print(f"  [WARN] No photos in {folder_name}")
            continue

        folder_aug = 0
        for photo in photos:
            base  = Path(photo).stem
            count = augment_single_image(
                os.path.join(src_folder, photo),
                aug_folder,
                base,
                AUGMENTATIONS_PER_PHOTO
            )
            folder_aug += count

        total_in_folder = len(photos) + folder_aug
        print(f"  {folder_name}: {len(photos)} original "
              f"→ {total_in_folder} total images")
        total_created += folder_aug

    return total_created


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COLLECT LFW NEGATIVES
# ─────────────────────────────────────────────────────────────────────────────
def collect_lfw_negatives():
    neg_dir = os.path.join(AUGMENTED_DIR, "negatives")

    if not os.path.exists(LFW_DIR):
        print(f"  [WARN] LFW not found at: {LFW_DIR}")
        print("         Skipping. This is optional.")
        return 0

    os.makedirs(neg_dir, exist_ok=True)

    all_images = []
    for person in os.listdir(LFW_DIR):
        person_path = os.path.join(LFW_DIR, person)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(person_path, img_file))

    if not all_images:
        print("  [WARN] No images found in LFW directory.")
        return 0

    selected = random.sample(
        all_images, min(NUM_NEGATIVES, len(all_images))
    )

    for i, src in enumerate(selected):
        dst = os.path.join(neg_dir, f"lfw_negative_{i:04d}.jpg")
        shutil.copy2(src, dst)

    print(f"  Collected {len(selected)} LFW faces "
          f"→ data/augmented/negatives/")
    return len(selected)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — REBUILD EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────
def rebuild_embeddings():
    """
    Temporarily overrides config paths so EmbeddingBuilder
    reads from augmented folder and saves to embeddings_augmented.
    Uses your exact pipeline with CLAHE and outlier removal.
    """
    original_raw = config.RAW_IMAGES_DIR
    original_emb = config.EMBEDDINGS_DIR

    config.RAW_IMAGES_DIR = AUGMENTED_DIR
    config.EMBEDDINGS_DIR = EMB_NEW_DIR
    os.makedirs(EMB_NEW_DIR, exist_ok=True)

    print(f"  Reading from : data/augmented/")
    print(f"  Saving to   : data/embeddings_augmented/")
    print()

    try:
        builder = EmbeddingBuilder()
        records = builder.build_database()
    except Exception as e:
        print(f"  [ERROR] EmbeddingBuilder failed: {e}")
        records = []
    finally:
        # Always restore config regardless of success or failure
        config.RAW_IMAGES_DIR = original_raw
        config.EMBEDDINGS_DIR = original_emb

    return records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — ACTIVATE NEW EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────
def activate_new_embeddings():
    """
    Copies names.txt then replaces data/embeddings/ with
    data/embeddings_augmented/
    """
    # Copy names.txt from backup into new embeddings folder
    names_copied = False
    for src_dir in [EMB_BACKUP_DIR, config.EMBEDDINGS_DIR]:
        names_src = os.path.join(src_dir, "names.txt")
        names_dst = os.path.join(EMB_NEW_DIR, "names.txt")
        if os.path.exists(names_src) and not os.path.exists(names_dst):
            shutil.copy2(names_src, names_dst)
            print(f"  Copied names.txt from {os.path.basename(src_dir)}/")
            names_copied = True
            break

    if not names_copied:
        print("  [WARN] names.txt not found in backup. "
              "Student names may show as IDs only.")

    # Replace active embeddings
    if os.path.exists(config.EMBEDDINGS_DIR):
        shutil.rmtree(config.EMBEDDINGS_DIR)
    shutil.copytree(EMB_NEW_DIR, config.EMBEDDINGS_DIR)

    print(f"  New embeddings active in: data/embeddings/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  NIGRAAN — Augmentation and Embedding Rebuild")
    print("=" * 62)

    # Step 1
    print("\n[STEP 1] Backing up existing embeddings...")
    backup_old_embeddings()

    # Step 2
    print("\n[STEP 2] Augmenting student photos...")
    total_aug = augment_all_students()

    if total_aug == 0:
        print("\n  [ERROR] No augmented images created.")
        print("  Check that data/raw_images/ has student folders with photos.")
        return

    print(f"\n  Total augmented images created: {total_aug}")

    # Step 3
    print("\n[STEP 3] Collecting LFW negative samples...")
    collect_lfw_negatives()

    # Step 4
    print("\n[STEP 4] Rebuilding embeddings from augmented photos...")
    print("  Using your EmbeddingBuilder with CLAHE + outlier removal.\n")
    records = rebuild_embeddings()

    if not records:
        print("\n  [ERROR] No embeddings built.")
        print("  Restoring original embeddings from backup...")
        if os.path.exists(EMB_BACKUP_DIR):
            if os.path.exists(config.EMBEDDINGS_DIR):
                shutil.rmtree(config.EMBEDDINGS_DIR)
            shutil.copytree(EMB_BACKUP_DIR, config.EMBEDDINGS_DIR)
            print("  Original embeddings restored successfully.")
        return

    # Step 5
    print("\n[STEP 5] Activating new embeddings...")
    activate_new_embeddings()

    # Summary
    print("\n" + "=" * 62)
    print("  DONE")
    print("=" * 62)
    print(f"\n  Students processed: {len(records)}")
    for r in records:
        print(f"    {r['student_id']} — {r['name']}")

    print("\n  Folder changes:")
    print("  data/embeddings/           → new improved embeddings (active)")
    print("  data/embeddings_backup/    → your original (safe)")
    print("  data/augmented/            → all augmented photos")
    print("  data/embeddings_augmented/ → new embeddings pre-activation")
    print("\n  Run your system:")
    print("    python main.py")
    print("\n  If worse, restore with:")
    print("    python restore_embeddings.py")
    print("=" * 62)


if __name__ == "__main__":
    main()