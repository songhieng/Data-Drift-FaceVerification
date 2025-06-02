Below is a brief discussion of test-time data augmentation (TTA) applied at the feature-extraction stage for a face-verification system, followed by a modified version of your benchmark script that implements TTA for DeepFace’s FaceNet512. The idea is to generate a few augmented variants of each input face (e.g., flips, brightness shifts), extract embeddings for each variant, average them into a “robust” embedding, and then proceed with the usual 10‐fold evaluation.

---

## 1. Test-Time Data Augmentation for Face Verification

### Would it improve accuracy? Why?

* **Yes, in many cases.** A pretrained embedding network (FaceNet512) was trained on “natural” faces. Real-world verification, however, encounters variations in pose, illumination, slight occlusions, etc. By applying simple augmentations (horizontal flip, small brightness/contrast shifts, Gaussian noise, minor rotations) at inference time and averaging the resulting embeddings, you effectively “denoise” any spurious effects caused by, say, uneven lighting or a small misalignment. This tends to push genuine‐pair similarities closer together and impostor similarities further apart, thereby reducing EER and boosting AUC/accuracy.

### Pros

* **Increased robustness.** The averaged embedding is less sensitive to outliers (e.g., a single bad lighting condition).
* **No retraining required.** You use the same pretrained FaceNet512 model; you only add inference‐time preprocessing.
* **Relatively easy to implement.** A handful of simple OpenCV transforms suffice.

### Cons

* **Higher inference cost.** If you sample 4 augmentations per image, you do 4× more forward passes → roughly 4× longer embedding time.
* **Diminishing returns.** Beyond 3–5 simple augmentations, the gains plateau (and might even degrade if augmentations become too extreme).
* **Possible domain mismatch.** If augmentations are unrealistic (e.g., heavy blurring), you might inject noise that confuses the model instead of helping.

---

## 2. Modified Benchmark Script with TTA (FaceNet512 + Augmentation)

Below is a self-contained `main.py`. It follows your original pipeline but replaces `get_df_embeddings` with a version that generates a few simple augmentations per image, computes embeddings, and averages them. In short:

1. For each original image:

   * Read with OpenCV.
   * Generate a small set of variants (original, horizontal flip, brightened, darkened).
   * Call `DeepFace.represent(img=...)` on each variant.
   * L2-normalize each embedding, then average and re‐normalize.
2. Everything else (pair loading, fold evaluation, CSV export) remains the same.

```python
#!/usr/bin/env python3
"""
main.py

Runs a 10‐fold evaluation (FF and FP protocols) on the CFP dataset using:
  - DeepFace’s Facenet512 embeddings (with test‐time augmentation)
  - InsightFace’s buffalo_l with a custom ONNX recognizer

Metrics per fold:
  - ACC (accuracy at the EER threshold)
  - AUC (area under the ROC curve)
  - EER (equal error rate)

Additionally, this script writes out every pair’s similarity score and label
into `all_similarity_results.csv` with columns:
    model,mode,fold,img1,img2,similarity,label

Usage:
  1. Put this script in the same folder where “cfp-dataset/cfp-dataset” resides.
  2. Install dependencies:
       pip install numpy scikit-learn deepface insightface onnxruntime pandas opencv-python
  3. Run:
       python main.py
"""

import os
import time
import cv2
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from deepface import DeepFace
import insightface
from insightface.app import FaceAnalysis

# ------------------------------------------------------------------------------

def load_map(file_path, base_folder):
    """
    Reads a Pair_list file (index and relative path) and returns a dict:
        { index → absolute_image_path }.
    """
    m = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, rel_path = line.split()
            idx = int(idx_str)
            abs_path = os.path.normpath(os.path.join(base_folder, rel_path))
            m[idx] = abs_path
    return m

def augment_variants(bgr_img):
    """
    Given an OpenCV BGR image, return a list of augmented BGR images:
      - original
      - horizontally flipped
      - brightness +30
      - brightness -30
    (Clamp pixel values to [0,255].)
    """
    variants = []

    # (1) Original
    variants.append(bgr_img)

    # (2) Horizontal flip
    variants.append(cv2.flip(bgr_img, 1))

    # (3) Brightness +30
    bright_plus = cv2.convertScaleAbs(bgr_img, alpha=1.0, beta=30)
    variants.append(bright_plus)

    # (4) Brightness -30
    bright_minus = cv2.convertScaleAbs(bgr_img, alpha=1.0, beta=-30)
    variants.append(bright_minus)

    return variants

def get_df_embeddings_tta(all_image_paths, tta_batch_size=4, enforce_detection=False):
    """
    For each image path in all_image_paths:
      - Read via OpenCV (BGR).
      - Generate augmentations (4 total).
      - For each variant, call DeepFace.represent(img=..., model_name='Facenet512').
      - Normalize each embedding, average them, then L2-normalize the final vector.
    Returns dict {img_path → averaged_embedding}.
    """
    embeddings = {}
    total = len(all_image_paths)
    for idx, img_path in enumerate(all_image_paths, start=1):
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  ⚠️ Could not read {img_path} (skipping)")
            continue

        variants = augment_variants(bgr)
        emb_list = []

        for var in variants:
            # DeepFace.represent accepts either img_path= or img= (numpy array in BGR)
            try:
                rep = DeepFace.represent(
                    img=var,
                    model_name='Facenet512',
                    enforce_detection=enforce_detection
                )
            except Exception as e:
                print(f"    ⚠️ DeepFace error on variant of {img_path}: {str(e)}")
                continue

            # rep might be list-of-dict or dict
            if isinstance(rep, list) and isinstance(rep[0], dict) and "embedding" in rep[0]:
                vec = np.array(rep[0]["embedding"], dtype=np.float32)
            elif isinstance(rep, dict) and "embedding" in rep:
                vec = np.array(rep["embedding"], dtype=np.float32)
            else:
                print(f"    ⚠️ Unexpected DeepFace output for {img_path}")
                continue

            # L2 normalize each variant embedding
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                emb_list.append(vec)

        if not emb_list:
            continue

        # Average and re-normalize
        avg_emb = np.mean(np.stack(emb_list, axis=0), axis=0)
        avg_emb /= np.linalg.norm(avg_emb)
        embeddings[img_path] = avg_emb

        if idx % 200 == 0 or idx == total:
            print(f"  [TTA] Processed {idx}/{total} images")

    return embeddings

def get_if_embeddings(all_image_paths, insight_app, print_every=200):
    """
    Given a sorted list of image paths, loads each image via OpenCV and runs
    InsightFace.app.FaceAnalysis.get(...) to extract embeddings. Prints progress.

    Returns: dict { img_path → normalized_embedding }.
    """
    embeddings = {}
    total = len(all_image_paths)
    print(f"    [InsightFace] Embedding {total} images ...")
    start_time = time.time()

    for idx, img_path in enumerate(all_image_paths, start=1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"      ⚠️ Could not read {img_path} (skipping)")
            continue

        try:
            faces = insight_app.get(img_bgr)
        except Exception as e:
            print(f"      ⚠️ Skipped {img_path} (InsightFace error: {str(e)})")
            continue

        if not faces:
            continue

        emb_vec = faces[0].embedding.astype(np.float32)
        emb_vec /= np.linalg.norm(emb_vec)
        embeddings[img_path] = emb_vec

        if idx % print_every == 0 or idx == total:
            elapsed = time.time() - start_time
            print(f"      [InsightFace] {idx}/{total} done  (elapsed {elapsed:.1f}s)")

    print(f"    [InsightFace] Completed embeddings: {len(embeddings)}/{total}")
    return embeddings

def load_pairs_for_fold(split_dir, maps_tuple):
    """
    For a given split folder (e.g., .../Split/FF/1), read same.txt and diff.txt,
    then map indices → absolute paths using maps_tuple = (mapA, mapB).
    Returns two lists of (path1, path2):
      same_pairs = [ (pathA_i, pathA_j), ... ]
      diff_pairs = [ (pathA_i, pathB_j), ... ]
    """
    same = []
    diff = []
    for filename, container in [('same.txt', same), ('diff.txt', diff)]:
        full_path = os.path.join(split_dir, filename)
        with open(full_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                i1_str, i2_str = line.split(',')
                i1, i2 = int(i1_str), int(i2_str)
                p1 = maps_tuple[0][i1]
                p2 = maps_tuple[1][i2]
                container.append((p1, p2))
    return same, diff

def evaluate_fold(model_embs, same_pairs, diff_pairs):
    """
    Given a dict of embeddings {img_path→vector}, plus two lists of pairs,
    compute:
      - sims & labels arrays,
      - ACC, AUC, EER
    Returns (acc, auc_score, eer, sims_array, labels_array, pair_list)
      where pair_list is [(img1,img2), ...] in the same order as sims & labels.
    """
    sims = []
    labels = []
    pair_list = []

    # Process "same" pairs with label=1
    for (p1, p2) in same_pairs:
        if p1 not in model_embs or p2 not in model_embs:
            continue
        sim_val = float(np.dot(model_embs[p1], model_embs[p2]))
        sims.append(sim_val)
        labels.append(1)
        pair_list.append((p1, p2))

    # Process "diff" pairs with label=0
    for (p1, p2) in diff_pairs:
        if p1 not in model_embs or p2 not in model_embs:
            continue
        sim_val = float(np.dot(model_embs[p1], model_embs[p2]))
        sims.append(sim_val)
        labels.append(0)
        pair_list.append((p1, p2))

    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # If there are no valid pairs, return zeros
    if labels.size == 0:
        return 0.0, 0.0, 0.0, sims, labels, pair_list

    fpr, tpr, thresholds = roc_curve(labels, sims, pos_label=1)
    auc_score = auc(fpr, tpr)

    fnr = 1.0 - tpr
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0
    thr_eer = thresholds[idx_eer]

    preds = (sims >= thr_eer).astype(np.int32)
    acc = accuracy_score(labels, preds)
    return acc, auc_score, eer, sims, labels, pair_list

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # (A) Locate the “cfp-dataset” folder and protocol subfolders
    base = "cfp-dataset"
    if not os.path.isdir(os.path.join(base, "Protocol")):
        nested = os.path.join(base, os.path.basename(base))
        if os.path.isdir(os.path.join(nested, "Protocol")):
            base = nested
    protocol_dir = os.path.join(base, "Protocol")

    # (B) Build index→path maps for frontal (F) and profile (P)
    frontal_map = load_map(
        os.path.join(protocol_dir, "Pair_list_F.txt"),
        protocol_dir
    )
    profile_map = load_map(
        os.path.join(protocol_dir, "Pair_list_P.txt"),
        protocol_dir
    )

    # (C) Gather all distinct image paths from all 10 folds of both modes
    all_paths = set()
    for mode_name, maps_tuple in [("FF", (frontal_map, frontal_map)),
                                  ("FP", (frontal_map, profile_map))]:
        split_folder = os.path.join(protocol_dir, "Split", mode_name)
        for fold_name in sorted(os.listdir(split_folder), key=lambda x: int(x)):
            fold_dir = os.path.join(split_folder, fold_name)
            same_pairs, diff_pairs = load_pairs_for_fold(fold_dir, maps_tuple)
            for (pA, pB) in (same_pairs + diff_pairs):
                all_paths.add(pA)
                all_paths.add(pB)

    all_paths = sorted(all_paths)
    print(f"Total distinct images to embed: {len(all_paths)}")

    # (D) Extract DeepFace (Facenet512) embeddings with TTA
    print("Extracting DeepFace (Facenet512) embeddings with TTA...")
    df_embeddings = get_df_embeddings_tta(all_paths, enforce_detection=False)
    print(f"  → Done: {len(df_embeddings)} images embedded by DeepFace (with TTA).")

    # (E) Initialize InsightFace buffalo_l and swap in custom ONNX
    print("Initializing InsightFace buffalo_l and swapping in custom ONNX…")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(288, 288))

    print("Loading custom ONNX recognizer…")
    custom_recognizer = insightface.model_zoo.get_model('model.onnx')
    custom_recognizer.prepare(ctx_id=0)
    # Replace the “recognition” head with our custom ONNX
    for model_name, model_instance in app.models.items():
        if model_instance.taskname == 'recognition':
            app.models[model_name] = custom_recognizer
            break

    # (F) Extract InsightFace (custom ONNX) embeddings
    print("Extracting InsightFace (custom ONNX) embeddings...")
    if_embeddings = get_if_embeddings(all_paths, app)
    print(f"  → Done: {len(if_embeddings)} images embedded by InsightFace.")

    # (G) Prepare to write every similarity score into a CSV
    csv_filename = "all_similarity_results.csv"
    csv_fields = ["model", "mode", "fold", "img1", "img2", "similarity", "label"]
    csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # (H) Run 10‐fold evaluation for both modes and both models,
    #     saving per‐pair similarities to CSV as we go.
    results = {'deepface_tta': [], 'insight': []}
    for mode_name, maps_tuple in [
        ("FF", (frontal_map, frontal_map)),
        ("FP", (frontal_map, profile_map))
    ]:
        print(f"\nRunning 10‐fold evaluation on mode = '{mode_name}' …")
        split_folder = os.path.join(protocol_dir, "Split", mode_name)

        for fold_name in sorted(os.listdir(split_folder), key=lambda x: int(x)):
            fold_dir = os.path.join(split_folder, fold_name)
            same_pairs, diff_pairs = load_pairs_for_fold(fold_dir, maps_tuple)

            # —— DeepFace (TTA) fold evaluation + CSV export —— #
            acc_df, auc_df, eer_df, sims_df, labels_df, pairs_df = evaluate_fold(
                df_embeddings, same_pairs, diff_pairs
            )
            for ((img1, img2), sim_val, lbl) in zip(pairs_df, sims_df, labels_df):
                writer.writerow({
                    "model": "deepface_tta",
                    "mode": mode_name,
                    "fold": fold_name,
                    "img1": img1,
                    "img2": img2,
                    "similarity": f"{sim_val:.6f}",
                    "label": int(lbl)
                })

            # —— InsightFace fold evaluation + CSV export —— #
            acc_if, auc_if, eer_if, sims_if, labels_if, pairs_if = evaluate_fold(
                if_embeddings, same_pairs, diff_pairs
            )
            for ((img1, img2), sim_val, lbl) in zip(pairs_if, sims_if, labels_if):
                writer.writerow({
                    "model": "insight",
                    "mode": mode_name,
                    "fold": fold_name,
                    "img1": img1,
                    "img2": img2,
                    "similarity": f"{sim_val:.6f}",
                    "label": int(lbl)
                })

            results['deepface_tta'].append((acc_df, auc_df, eer_df))
            results['insight'].append((acc_if, auc_if, eer_if))

            print(
                f"  • Fold {fold_name:>2}: "
                f"DeepFace‐TTA → ACC={acc_df:.4f}, AUC={auc_df:.4f}, EER={eer_df:.4f} | "
                f"Insight → ACC={acc_if:.4f}, AUC={auc_if:.4f}, EER={eer_if:.4f}"
            )

    csv_file.close()
    print(f"\nSaved all pairwise similarities to '{csv_filename}'.\n")

    # (I) Compute and print mean ± std across all 20 folds
    def summarize(metric_list):
        arr = np.array(metric_list, dtype=np.float32)
        return float(np.mean(arr)), float(np.std(arr))

    for model_key, display_name in [('deepface_tta', 'DeepFace-TTA'),
                                    ('insight', 'InsightFace')]:
        arr = np.array(results[model_key], dtype=np.float32)  # shape: (20,3)
        accs = arr[:, 0]
        aucs = arr[:, 1]
        eers = arr[:, 2]
        m_acc, s_acc = summarize(accs)
        m_auc, s_auc = summarize(aucs)
        m_eer, s_eer = summarize(eers)
        print(f"{display_name} SUMMARY (across all 20 folds):")
        print(f"  → ACC = {m_acc:.4f} ± {s_acc:.4f}")
        print(f"  → AUC = {m_auc:.4f} ± {s_auc:.4f}")
        print(f"  → EER = {m_eer:.4f} ± {s_eer:.4f}")

    print("\nDone.")
```

**Key changes compared to your original script:**

1. **Function `augment_variants(bgr_img)`** generates four simple versions of each face:

   * Original
   * Horizontally flipped
   * Brightness +30
   * Brightness −30

2. **Function `get_df_embeddings_tta(...)`** replaces `get_df_embeddings(...)`. For each image path, it:

   * Reads the image once.
   * Builds its 4 variants.
   * Calls `DeepFace.represent(img=variant, model_name='Facenet512')` on each variant (skip if `DeepFace` fails or if detection is enforced and no face is found).
   * L2-normalizes each variant’s embedding, averages them, then re‐L2-normalizes.
   * Stores the result in `embeddings[img_path]`.

3. Everywhere else (pair loading, evaluation, CSV writing, InsightFace extraction) remains identical, except that the DeepFace model is now labeled `"deepface_tta"` in CSV and result tables.

---

With this setup, you can compare:

* **Baseline (original code)**: single‐shot embedding per image.
* **TTA version (above)**: 4 embeddings per image averaged.

In most cases, you should see a small but consistent drop in EER and a boost in AUC/accuracy—especially on “hard” pairs with lighting or pose mismatch. Just keep in mind that the runtime will be roughly 4× slower during the DeepFace embedding phase.

Feel free to tweak `augment_variants(...)` (e.g., add small rotations, Gaussian blur/noise) or the number of variants. Too many, or too‐harsh transforms, can backfire; typically 3–5 mild augmentations strike the best balance.
