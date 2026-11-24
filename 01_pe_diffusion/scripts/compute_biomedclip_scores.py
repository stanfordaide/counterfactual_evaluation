import os
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
import nibabel as nib
import open_clip
from PIL import Image
from tqdm import tqdm
import argparse
import time
import glob
import torch.nn.functional as F
import logging
import sys

# ------------------------------------------------------------------------------
# Create logger
# ------------------------------------------------------------------------------
logger = logging.getLogger("biomedclip_scores")
if not logger.handlers:
    # Configure logger only if it has no handlers (avoid reconfiguring in multi-rank scenarios)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)

def load_all_direction_slices(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    data = (data - data.min()) / (data.max() - data.min() + 1e-5)
    axial = [data[:, :, i] for i in range(data.shape[2])]
    coronal = [data[:, i, :] for i in range(data.shape[1])]
    sagittal = [data[i, :, :] for i in range(data.shape[0])]
    return axial + coronal + sagittal

# I want to load 64 slices before and after the center slice
def load_all_direction_slices_128(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    # I want to load 128 slices before and after the center slice
    center_slice = data.shape[2] // 2
    axial = [data[:, :, i] for i in range(center_slice - 128, center_slice + 128)]
    coronal = [data[:, i, :] for i in range(center_slice - 128, center_slice + 128)]
    sagittal = [data[i, :, :] for i in range(center_slice - 128, center_slice + 128)]
    return axial + coronal + sagittal

def slice_to_rgb_pil(slice_2d: np.ndarray) -> Image.Image:
    slice_uint8 = (slice_2d * 255).clip(0, 255).astype(np.uint8)
    rgb_array = np.stack([slice_uint8] * 3, axis=-1)
    return Image.fromarray(rgb_array)

def is_blank_slice(slice_2d, threshold=0.005):
    return slice_2d.std() < threshold

def compute_clipscore_biomedclip(images, text, model, preprocess, tokenizer, device, batch_size=32):
    scores = []
    tokens = tokenizer([text]).to(device)

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        batch_tensors = [preprocess(img).to(device) for img in batch_imgs]
        image_input = torch.stack(batch_tensors)

        with torch.no_grad():
            image_features, text_features, logit_scale = model(image_input, tokens)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarities = (logit_scale * image_features @ text_features.T).squeeze().clamp(min=0)
            scores.extend(similarities.tolist())

    return float(np.mean(scores)) if scores else None

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--csv_path", default="/path/to/inspect2/pe_training_data_all/inference_results_unet_3d_text_all_l2_loss_20250630_205622.csv")
    # parser.add_argument("--csv_path", default="/path/to/inspect2/pe_training_data_all/predictions/unet_3d_text_all_l2_loss_best_demo/inference_results_unet_3d_text_all_l2_loss_best_demo_20250730_155945.csv")
    # parser.add_argument("--csv_path", default="/path/to/inspect2/pe_training_data_all/predictions_controlnet_demo/controlnet_demo_predictions.csv")
    parser.add_argument("--csv_path", default="/path/to/inspect2/pe_training_data_all/real_demo_impressions_filenames.csv")
    parser.add_argument("--output_dir", default="biomedclip_scores_real")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--single_gpu", action="store_true")
    args = parser.parse_args()

    if args.single_gpu:
        local_rank = 0
        world_size = 1
        device = "cuda:0"
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = f"cuda:{local_rank}"

    # All CUDA and model logic happens after setting device
    model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, preprocess = open_clip.create_model_from_pretrained(model_id, device=device)
    tokenizer = open_clip.get_tokenizer(model_id, truncate=True, context_length=256)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"clipscore_rank{local_rank}.csv")

    df = pd.read_csv(args.csv_path)
    chunk = df if args.single_gpu else np.array_split(df, world_size)[local_rank]

    results = []
    for idx, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Rank {local_rank}"):
        # If I call debug with multiple GPUs, it will only run on Rank 0
        if args.debug and idx >= 5:
            break
        try:
            slices = load_all_direction_slices(row["prediction_filename"])
            slices = [s for s in slices if not is_blank_slice(s)]
            images = [slice_to_rgb_pil(s) for s in slices]

            impression = row["impression"]
            if pd.isna(impression):
                impression = ""
            elif isinstance(impression, np.ndarray):
                impression = " ".join(str(x) for x in impression.flatten())
            else:
                impression = str(impression)

            score = compute_clipscore_biomedclip(images, impression, model, preprocess, tokenizer, device, args.batch_size)
            logger.info(f"score: {score}")

            results.append({"image_id": row["image_id"], "clip_score": score})

        except Exception as e:
            logger.info(f"[RANK {local_rank} ERROR] {row.get('image_id', f'row_{idx}')}: {e}")

            results.append({
                "image_id": row.get("image_id", f"row_{idx}"),
                "clip_score": None,
                "error": str(e)
            })

    pd.DataFrame(results).to_csv(output_path, index=False)

    if not args.single_gpu:
        dist.barrier(device_ids=[local_rank])
        if local_rank == 0:
            time.sleep(2)
            all_parts = glob.glob(os.path.join(args.output_dir, "clipscore_rank*.csv"))
            df_all = pd.concat([pd.read_csv(f) for f in all_parts])
            df_all.to_csv(os.path.join(args.output_dir, "clipscore_merged.csv"), index=False)
            logger.info("[Rank 0] Merged results saved.")

if __name__ == "__main__":
    main()