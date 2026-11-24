#!/usr/bin/env python3
"""
FAST segmentation-only registration (crop + downsample + rigid/affine) and ICH pasting.

Input: a JSON file (--json_path) with a "training" list. Each entry gives (NIfTI):
  src_healthy_seg_path : healthy multi-class segmentation (recipient)
  src_doner_seg_path   : donor multi-class segmentation (same label scheme)
  src_ich_seg_path     : donor ICH binary mask (0/1, where 1 = ICH)
  segmentation         : output path: healthy segmentation with ICH pasted (label 66)

Key labels:
  skull = 23
  head  = 24 (optional to include in registration driver)
  ICH in donor_ich = 1
  ICH in output = 66
  brain structures: assumed 1..16 (0 assumed background; configurable)

No CT is used or modified.
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import nibabel as nib

import json
import logging
from tqdm import tqdm


# ----------------------------
# NIfTI header fix (ITK needs orthonormal direction)
# ----------------------------
def sanitize_nifti_affine(in_path: str, out_path: str):
    """
    Make NIfTI affine ITK-friendly (orthonormal direction cosines).
    - Reorients to closest canonical
    - Orthonormalizes rotation (removes shear) while keeping voxel sizes
    - Keeps translation
    """
    img = nib.load(in_path)
    img = nib.as_closest_canonical(img)

    data = img.get_fdata().astype(img.get_data_dtype())

    A = img.affine.copy()
    RZS = A[:3, :3]
    t = A[:3, 3].copy()

    zooms = img.header.get_zooms()[:3]
    S = np.diag(zooms)

    R_approx = RZS @ np.linalg.inv(S)
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    new_affine = np.eye(4, dtype=float)
    new_affine[:3, :3] = R @ S
    new_affine[:3, 3] = t

    new_img = nib.Nifti1Image(data, new_affine)
    new_img.set_qform(new_affine, code=1)
    new_img.set_sform(new_affine, code=1)
    new_img.header.set_zooms(zooms)
    nib.save(new_img, out_path)


# ----------------------------
# SimpleITK helpers
# ----------------------------
def read_img(path: str, pixel_type) -> sitk.Image:
    return sitk.ReadImage(path, pixel_type)

def make_binary_mask(label_img: sitk.Image, keep_values) -> sitk.Image:
    arr = sitk.GetArrayFromImage(label_img)
    mask = np.isin(arr, np.array(list(keep_values), dtype=arr.dtype)).astype(np.uint8)
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(label_img)
    return out

def signed_distance(mask_u8: sitk.Image) -> sitk.Image:
    return sitk.SignedMaurerDistanceMap(mask_u8, insideIsPositive=True,
                                        squaredDistance=False, useImageSpacing=True)

def resample_isotropic(img: sitk.Image,
                       out_spacing=(6.0, 6.0, 6.0),
                       interp=sitk.sitkLinear,
                       default_value=0) -> sitk.Image:
    in_spacing = np.array(img.GetSpacing(), dtype=float)
    in_size = np.array(img.GetSize(), dtype=float)
    out_spacing = np.array(out_spacing, dtype=float)

    out_size = np.maximum(1, np.round(in_size * (in_spacing / out_spacing))).astype(int)

    ref = sitk.Image([int(x) for x in out_size], img.GetPixelID())
    ref.SetSpacing([float(x) for x in out_spacing])
    ref.SetDirection(img.GetDirection())
    ref.SetOrigin(img.GetOrigin())

    return sitk.Resample(img, ref, sitk.Transform(3, sitk.sitkIdentity),
                         interp, default_value)

def mask_bbox_index(mask_u8: sitk.Image, margin_mm=25.0):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    if not stats.HasLabel(1):
        raise RuntimeError("Mask is empty; cannot compute bbox for cropping.")
    bb = stats.GetBoundingBox(1)  # (x,y,z, sx,sy,sz)
    start = np.array(bb[:3], dtype=int)
    size = np.array(bb[3:], dtype=int)

    spacing = np.array(mask_u8.GetSpacing(), dtype=float)
    margin_vox = np.ceil(margin_mm / spacing).astype(int)

    start2 = np.maximum(0, start - margin_vox)
    end2 = np.minimum(np.array(mask_u8.GetSize(), dtype=int), start + size + margin_vox)
    size2 = (end2 - start2).astype(int)

    return tuple(start2.tolist()), tuple(size2.tolist())

def crop_roi(img: sitk.Image, start, size) -> sitk.Image:
    return sitk.RegionOfInterest(img, size=size, index=start)

def resample_label(moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform,
                   default_value=0, out_type=sitk.sitkUInt8) -> sitk.Image:
    return sitk.Resample(moving, reference, transform, sitk.sitkNearestNeighbor,
                         default_value, out_type)


# ----------------------------
# Fast registration on distance maps (rigid / affine)
# ----------------------------
def fast_register(fixed_dm: sitk.Image,
                  moving_dm: sitk.Image,
                  do_affine: bool,
                  iters: int = 60) -> sitk.Transform:
    """
    Returns transform mapping moving -> fixed in physical space.
    """
    init_rigid = sitk.CenteredTransformInitializer(
        fixed_dm, moving_dm,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Rigid
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3,
        numberOfIterations=iters,
        gradientMagnitudeTolerance=1e-6
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([2, 1])
    R.SetSmoothingSigmasPerLevel([1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R.SetInitialTransform(init_rigid, inPlace=False)
    rigid_out = R.Execute(fixed_dm, moving_dm)

    if not do_affine:
        return rigid_out

    # Affine refine (initialize from rigid safely)
    aff0 = sitk.AffineTransform(3)
    aff0.SetCenter(rigid_out.GetCenter())
    aff0.SetTranslation(rigid_out.GetTranslation())
    aff0.SetMatrix(rigid_out.GetMatrix())

    R2 = sitk.ImageRegistrationMethod()
    R2.SetMetricAsMeanSquares()
    R2.SetInterpolator(sitk.sitkLinear)
    R2.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4,
        numberOfIterations=max(40, iters),
        gradientMagnitudeTolerance=1e-6
    )
    R2.SetOptimizerScalesFromPhysicalShift()
    R2.SetShrinkFactorsPerLevel([2, 1])
    R2.SetSmoothingSigmasPerLevel([1, 0])
    R2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R2.SetInitialTransform(aff0, inPlace=False)
    affine_out = R2.Execute(fixed_dm, moving_dm)
    return affine_out


def generate_couterfactual_seg(healthy_seg_path, src_doner_seg_path, src_ich_seg_path, output_path, temp_dir, args):
    
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    # # Sanitize headers (canonical + orthonormal direction)

    heathy_filename = os.path.basename(healthy_seg_path).replace(".nii.gz", "_sanitized.nii.gz")
    donor_filename = os.path.basename(src_doner_seg_path).replace(".nii.gz", "_brain_sanitized.nii.gz")
    ich_filename = os.path.basename(src_ich_seg_path).replace(".nii.gz", "_ich__sanitized.nii.gz")

    healthy_fix = os.path.join(temp_dir, heathy_filename)
    donor_fix   = os.path.join(temp_dir, donor_filename)
    ich_fix     =  os.path.join(temp_dir, ich_filename)

    sanitize_nifti_affine(healthy_seg_path, healthy_fix)
    sanitize_nifti_affine(src_doner_seg_path, donor_fix)
    sanitize_nifti_affine(src_ich_seg_path, ich_fix)

    fixed_seg  = read_img(healthy_fix, sitk.sitkUInt16)
    moving_seg = read_img(donor_fix, sitk.sitkUInt16)
    moving_ich = read_img(ich_fix, sitk.sitkUInt8)  # binary/0-1 is fine



    driver_labels = {args.skull_label, args.head_label} if args.use_head_with_skull else {args.skull_label}
    brain_labels = set(range(args.brain_start, args.brain_end + 1))

    fixed_skull  = make_binary_mask(fixed_seg, driver_labels)
    moving_skull = make_binary_mask(moving_seg, driver_labels)
    fixed_brain  = make_binary_mask(fixed_seg, brain_labels)

    # Crop each volume by its own skull bbox (safe for different sizes/origins)
    f_start, f_size = mask_bbox_index(fixed_skull, margin_mm=args.margin_mm)
    m_start, m_size = mask_bbox_index(moving_skull, margin_mm=args.margin_mm)

    fixed_skull_c  = crop_roi(fixed_skull, f_start, f_size)
    moving_skull_c = crop_roi(moving_skull, m_start, m_size)

    fixed_dm  = signed_distance(fixed_skull_c)
    moving_dm = signed_distance(moving_skull_c)

    sp = (args.reg_spacing_mm, args.reg_spacing_mm, args.reg_spacing_mm)
    fixed_dm_ds  = resample_isotropic(fixed_dm,  sp, sitk.sitkLinear, 0)
    moving_dm_ds = resample_isotropic(moving_dm, sp, sitk.sitkLinear, 0)

    final_tx = fast_register(fixed_dm_ds, moving_dm_ds, do_affine=args.affine, iters=args.iters)

    # Donor ICH is 0/1; build binary mask of ICH==1
    ich_arr = sitk.GetArrayFromImage(moving_ich)
    ich_bin = (ich_arr == args.ich_in_label).astype(np.uint8)
    ich_bin = sitk.GetImageFromArray(ich_bin)
    ich_bin.CopyInformation(moving_ich)

    # Warp into healthy seg grid
    ich_in_fixed = resample_label(ich_bin, fixed_seg, final_tx, default_value=0, out_type=sitk.sitkUInt8)

    # Clip to healthy brain and merge (write as 66)
    ich_in_fixed = sitk.And(ich_in_fixed, sitk.Cast(fixed_brain, sitk.sitkUInt8))

    fixed_arr = sitk.GetArrayFromImage(fixed_seg).astype(np.int32)
    ich_arr_f = sitk.GetArrayFromImage(ich_in_fixed).astype(np.uint8)

    fixed_arr[ich_arr_f > 0] = args.ich_out_label

    out = sitk.GetImageFromArray(fixed_arr.astype(np.uint16))
    out.CopyInformation(fixed_seg)
    sitk.WriteImage(out, output_path)



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", default="/path/to/ich_pairs.json")
    ap.add_argument("--temp_dir", default="/path/to/temp_dir")
    ap.add_argument("--skull_label", type=int, default=23)
    ap.add_argument("--head_label", type=int, default=24)
    ap.add_argument("--ich_in_label", type=int, default=1, help="ICH label in donor_ich (default 1)")
    ap.add_argument("--ich_out_label", type=int, default=66, help="ICH label to write into output (default 66)")
    ap.add_argument("--brain_start", type=int, default=1)
    ap.add_argument("--brain_end", type=int, default=16)
    ap.add_argument("--use_head_with_skull", action="store_true")
    ap.add_argument("--margin_mm", type=float, default=25.0)
    ap.add_argument("--reg_spacing_mm", type=float, default=6.0, help="Try 8 for more speed")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--affine", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.temp_dir, exist_ok=True)

    print(f"Processing json path: {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)["training"]
    logging.info(f"Total number of samples in the data list: {len(data_list)}")

    for idx, data in tqdm(enumerate(data_list), total=len(data_list), desc="Processing samples"):
        healthy_seg_path = data["src_healthy_seg_path"]
        src_doner_seg_path = data["src_doner_seg_path"]
        src_ich_seg_path = data["src_ich_seg_path"]
        output_path = data["segmentation"]
        logging.info(f"[{idx+1}/{len(data_list)}] Processing: {output_path} ...")

        generate_couterfactual_seg(healthy_seg_path, src_doner_seg_path, src_ich_seg_path,
                                output_path, args.temp_dir, args)




