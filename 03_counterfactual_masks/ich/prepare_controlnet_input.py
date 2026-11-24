#!/usr/bin/env python3
"""Build ControlNet input JSON from generated ICH counterfactual segmentations.

Inputs: the pairing JSON (--pairs_json), the donor table (--ich_info_scv) and a
private table of the source scans (--patient_info_csv, not released; columns
patient_id, impression, age, gender, race). The pairing JSON must already exist
and its "segmentation" masks must already be generated; this script does not
register masks. Output: controlnet_input.json next to the pairing JSON.
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import re

ICH_NAMES = {"SDH": "subdural", "IPH": "intraparenchymal",
             "IVH": "intraventricular", "SAH": "subarachnoid", "EDH": "epidural"}


def load_ich_pairs_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["training"]


def modify_impression(original_impression, ich_subtype_string, ich_volume, age, race, gender):
    demo_info = str(age) + " years old " + race + " " + gender + "."

    modified_impression = remove_original_ich_info(original_impression)
    ich_subtypes = ich_subtype_string.split(";")## get the list of ich subtypes, remove the 
    ## remove the empty strings from the ich_subtypes list
    ## remove all the spaces from the ich_subtypes list
    ich_subtypes = [ich_subtype for ich_subtype in ich_subtypes if ich_subtype != " "]
    ich_subtypes = [ich_subtype.strip() for ich_subtype in ich_subtypes]
    ich_string = ""
    for ich_subtype in ich_subtypes:
        ich_string += ICH_NAMES[ich_subtype] + ", "
    new_ich_prompt = "1. There is " + ich_string.rstrip(", ") + " hemorrhage shown in the CT scan. With total volume of " + str(ich_volume) + " ml."
    new_text_prompt = modified_impression.replace("IMPRESSION:", demo_info+ " IMPRESSION: " + new_ich_prompt)
    return new_text_prompt


def remove_original_ich_info(original_impression):
    impression = original_impression
    match = re.search(r'1\.\s+(.+?)(?=\s*\d+\.|$)', impression)
    if match:
        result = "1.  " + match.group(1)
        impression = impression.replace(result, "")
    else:
        ## find the sentence "No acute intracranial hemorrhage or large vascular territory infarction."
        match = re.search(r'No acute intracranial hemorrhage or large vascular territory infarction.', impression)
        if match:
            result = "No acute intracranial hemorrhage or large vascular territory infarction."
            impression = impression.replace(result, "")

    return impression


def get_ich_volume(seg_path,tgt_shape=(512, 512, 512), tgt_spacing=(0.43, 0.43, 0.29), tgt_label=66):
    seg_nib = nib.load(seg_path)
    seg_data = seg_nib.get_fdata()
    shape = seg_data.shape
    if shape != tgt_shape:
        from scipy.ndimage import zoom
        zoom_factors = [tgt_shape[i] / seg_data.shape[i] for i in range(3)]
        seg_data = zoom(seg_data, zoom_factors, order=0)  # use nearest neighbor interpolation for segmentation
    ich_mask = (seg_data == tgt_label)
    tgt_voxel_size = tgt_spacing[0] * tgt_spacing[1] * tgt_spacing[2] / 1000.0
    volume = float(np.sum(ich_mask) * tgt_voxel_size)  # in ml
    ## only keep .2f of the volume
    volume = round(volume, 2)
    return volume



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs_json", default="/path/to/ich_pairs_2640.json")
    ap.add_argument("--patient_info_csv",
                    default=os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "ich", "patient_info.csv"))
    ap.add_argument("--ich_info_scv", default="/path/to/ich_info.csv")
    args = ap.parse_args()

    pairs = load_ich_pairs_json(args.pairs_json)
    print(f"Loaded {len(pairs)} ICH counterfactual pairs from {args.pairs_json}")

    ## get the patient information and the ich information

    patient_info = pd.read_csv(args.patient_info_csv)
    ich_info = pd.read_csv(args.ich_info_scv)

    controlnet_inputs = {"training": []}
    for item in pairs:
        seg_path = item["segmentation"]
        healthy_ich_seg_path = item["src_healthy_seg_path"]
        ich_doner_seg_path = item["src_doner_seg_path"]
        src_ich_seg_path = item["src_ich_seg_path"]
        patient_id = healthy_ich_seg_path.split("/")[-1].split(".nii.gz")[0]
        ich_id = src_ich_seg_path.split("/")[-1].split(".nii.gz")[0]
        item["image"] = seg_path ## controlnet use "image" field to create the output image filename.
        item["embedding"] = seg_path ## place holder, the data loader reads this key but controlnet does not use it.
        ## find the patient information from the patient_info dataframe
        patient_row = patient_info[patient_info["patient_id"] == patient_id]
        if len(patient_row) == 0:
            print(f"Patient ID {patient_id} not found in patient info CSV.")
            continue
        item["dim"] = [512, 512, 512]
        item["spacing"] = [0.43, 0.43, 0.29]
        
        
        original_impression = patient_row.iloc[0]["impression"]
        ## not necessary for controlnet, useful to analyze downstream tasks
        item["patient_id"] = patient_id ## not necessary for controlnet, useful to analyze downstream tasks
        item["age"] = int(patient_row.iloc[0]["age"])
        item["gender"] = str(patient_row.iloc[0]["gender"])
        item["race"] = str(patient_row.iloc[0]["race"])

        ## find the ich information from the ich_info dataframe
        ich_row = ich_info[ich_info["ich_idx"] == int(ich_id)]
        if len(ich_row) == 0:
            print(f"ICH ID {ich_id} not found in ICH info CSV.")
            continue
        item["ich_subtype"] = ich_row.iloc[0]["subtype"]

        ## calculate the ich volume based on the segmentation path        
        item["ich_volume_ml"] = get_ich_volume(seg_path)
        new_text_prompt = modify_impression(original_impression, item["ich_subtype"], item["ich_volume_ml"], item["age"], item["race"], item["gender"])
        item["text_prompt"] = new_text_prompt
        controlnet_inputs["training"].append(item)
        
       
    ## write out the controlnet_input_json
    output_json_file = os.path.join(os.path.dirname(os.path.abspath(args.pairs_json)), "controlnet_input.json")

    with open(output_json_file, "w") as f:
       json.dump(controlnet_inputs, f, indent=4)



if __name__ == "__main__":
    main()
