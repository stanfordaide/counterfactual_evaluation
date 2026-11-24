import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, generate_binary_structure
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import sys






def get_image_id(filename):
    return filename.split('_segmentation')[0]



def get_all_files(src_dir):
    all_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                image_id = get_image_id(file)
                full_path = os.path.join(root, image_id)
                relative_path = os.path.relpath(full_path, src_dir)
                all_files.append(relative_path)  # full path
    return all_files





###
# craniofacial_structures (#7, values from 0.0-7.0): mandible, teeth_lower, skull, head, sinus_maxillary, sinus_frontal, teeth_upper
# brain_structures (#16, values from 0.0 - 16.0): 
# brainstem, subarachnoid_space, venous_sinuses, septum_pellucidum, cerebellum, 
# caudate_nucleus, lentiform_nucleus, insular_cortex, internal_capsule, ventricle, 
# central_sulcus, frontal_lobe, parietal_lobe, occipital_lobe, temporal_lobe, thalamus. 

# 0.0 is background for both segmentations.

## I want to merge the two segmentations into one, keeping all labels. craniofacial structures will keep their original labels (0.0 - 7.0),
## brain structures will have their original labels (0 -17). So in the final merged segmentation,
## craniofacial structures will be in the range (0-7) + 20, i.e. (20-27)














# === Process each image ===

def process_image(rel_path):
    ## this function is used to merge the brain structure segmentation and craniofacial structure segmentation into one segmentation
    

    brain_structure_segmentation_dir = '/path/to/total_segmentator/total_segmentator_brain_structures'
    craniofacial_structures_segmentation_dir = '/path/to/total_segmentator/total_segmentator_craniofacial_structures'
    output_dir = '/path/to/total_segmentator/total_segmentator_brain_craniofacial_merged'
    
    
    os.makedirs(output_dir, exist_ok=True)


    try:
        # Check for rel_path in the output directory and only run the ones that are not there
        exit_file_path = os.path.join(output_dir, f"{rel_path}_segmentation_brain_craniofacial_merged.nii.gz")
        if os.path.exists(exit_file_path):
            return f"[{rel_path}] Skipped (already exists)."

        brain_structure_path = os.path.join(brain_structure_segmentation_dir, f"{rel_path}_segmentation_brain_structures.nii.gz")
        craniofacial_structures_path = os.path.join(craniofacial_structures_segmentation_dir, f"{rel_path}_segmentation_craniofacial_structures.nii.gz")

        brain_structure_nii = nib.load(brain_structure_path)
        craniofacial_structures_nii = nib.load(craniofacial_structures_path)

        brain_structure_data = brain_structure_nii.get_fdata().astype(np.uint16)
        craniofacial_structures_data = craniofacial_structures_nii.get_fdata().astype(np.uint16)

        if brain_structure_data.shape != craniofacial_structures_data.shape:
            return f"[{rel_path}] Skipped (shape mismatch)."
        
        # Step 0: Create smoothed and expanded 3D head mask
        struct = generate_binary_structure(3, 1)  # 3D connectivity
        ## get the head mask by selecting the labels in craniofacial segmentation that correspond to head, which is 4 and skull which is 3
        head_mask = ((craniofacial_structures_data == 4) | (craniofacial_structures_data == 3)).astype(int)
        head_mask = binary_closing(head_mask, structure=struct, iterations=1)
        head_mask = binary_dilation(head_mask, structure=struct, iterations=4)

        ## Step 1: Fill in the brain structures into the head mask
        filled_brain_in_head_mask = np.zeros_like(brain_structure_data)
        filled_brain_in_head_mask[head_mask == 1] = brain_structure_data[head_mask == 1]

        ## Step 2: Fill in the craniofacial structures into the head mask
        # for the other labels in craniofacial segmentation, add them to the filled_brain_in_head_mask, and change the labels by adding 20 to avoid overlap with brain structure labels
        for label in np.unique(craniofacial_structures_data):
            if label in [0, 4]:  # skip background, and head
                continue
            filled_brain_in_head_mask[(craniofacial_structures_data == label) & (head_mask == 1)] = label + 20
        
        ## Step 3: 
        ## fill in the background_mask regions in the filled_brain_and_craniofacial_in_head_mask with label 24 (which is head (4-->24))
        filled_brain_in_head_mask[(head_mask) & (filled_brain_in_head_mask == 0)] = 24



        # Save result
        out_path = os.path.join(output_dir, f"{rel_path}_segmentation_brain_craniofacial_merged.nii.gz")
        ## create the folder if not exists
        out_folder = os.path.dirname(out_path)
        os.makedirs(out_folder, exist_ok=True)
        merged_nii = nib.Nifti1Image(filled_brain_in_head_mask, affine=brain_structure_nii.affine, header=brain_structure_nii.header)
        nib.save(merged_nii, out_path)
        return f"[{rel_path}] Done"
    
    except Exception as e:
        return f"[{rel_path}] Error: {str(e)}"





def merge_ich_segmentations(brain_craniofacial_structures_data, cerebral_bleed_data):
    ## here we merge the cerebral bleed segmentation into the brain and craniofacial merged segmentation.
    ## previous segmentation already use values from 0~27, so here I use value 66 for cerebral bleed
     # Step 0: Sanity check
    if brain_craniofacial_structures_data.shape != cerebral_bleed_data.shape:
        raise ValueError("Input arrays must have the same shape.")
    merged = np.copy(brain_craniofacial_structures_data)
    ## for the celeral bleed regions (value == 1), set the merged label to 66, the other regions keep the original labels
    merged[cerebral_bleed_data == 1] = 66
    return merged



def process_image_add_ich(rel_path):
    ### this function is used to merge the brain_craniofacial_merged structure with cerebral bleed segmentation into one segmentation
    
    # Define paths


    cerebral_bleed_segmentation_dir = '/path/to/total_segmentator/total_segmentator_cerebral_bleed'
    brain_craniofacial_segmentation_dir = '/path/to/total_segmentator/total_segmentator_brain_craniofacial_merged'
    output_dir = '/path/to/total_segmentator/total_segmentator_brain_craniofacial_ICH_merged'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Check for rel_path in the output directory and only run the ones that are not there
        exit_file_path = os.path.join(output_dir, f"{rel_path}_segmentation_brain_craniofacial_ICH_merged.nii.gz")

        # if os.path.exists(exit_file_path):
        #     return f"[{rel_path}] Skipped (already exists)."

        
        cerebral_bleed_path = os.path.join(cerebral_bleed_segmentation_dir, f"{rel_path}_segmentation_cerebral_bleed.nii.gz")
        brain_craniofacial_structures_path = os.path.join(brain_craniofacial_segmentation_dir, f"{rel_path}_segmentation_brain_craniofacial_merged.nii.gz")

        cerebral_bleed_nii = nib.load(cerebral_bleed_path)
        brain_craniofacial_structures_nii = nib.load(brain_craniofacial_structures_path)

        cerebral_bleed_data = cerebral_bleed_nii.get_fdata().astype(np.uint16)
        brain_craniofacial_structures_data = brain_craniofacial_structures_nii.get_fdata().astype(np.uint16)

        if cerebral_bleed_data.shape != brain_craniofacial_structures_data.shape:
            print(f"Skipping {rel_path} due to shape mismatch: {cerebral_bleed_data.shape} vs {brain_craniofacial_structures_data.shape}")

        merged = merge_ich_segmentations(brain_craniofacial_structures_data, cerebral_bleed_data)

        # Save result
        out_path = os.path.join(output_dir, f"{rel_path}_segmentation_brain_craniofacial_ICH_merged.nii.gz")
        ## create the folder if not exists
        out_folder = os.path.dirname(out_path)
        os.makedirs(out_folder, exist_ok=True)
        merged_nii = nib.Nifti1Image(merged, affine=brain_craniofacial_structures_nii.affine, header=brain_craniofacial_structures_nii.header)
        nib.save(merged_nii, out_path)
        return f"[{rel_path}] Done"
    
    except Exception as e:
        return f"[{rel_path}] Error: {str(e)}"





# for rel_path in tqdm(image_paths):
#     print(process_image(rel_path))

# for rel_path in tqdm(image_paths):
#     print(process_image_add_ich(rel_path))



brain_structure_segmentation_dir = '/path/to/total_segmentator/total_segmentator_brain_structures'
craniofacial_structures_segmentation_dir = '/path/to/total_segmentator/total_segmentator_craniofacial_structures'
cerebral_bleed_segmentation_dir = '/path/to/total_segmentator/total_segmentator_cerebral_bleed'

brain_craniofacial_segmentation_dir = '/path/to/total_segmentator/total_segmentator_brain_craniofacial_merged'
brain_craniofacial_ICH_segmentation_dir = '/path/to/total_segmentator/total_segmentator_brain_craniofacial_ICH_merged'

if not os.path.exists(brain_craniofacial_ICH_segmentation_dir):
    os.makedirs(brain_craniofacial_ICH_segmentation_dir, exist_ok=True)
if not os.path.exists(brain_craniofacial_segmentation_dir):
    os.makedirs(brain_craniofacial_segmentation_dir, exist_ok=True)

task = "merge_brain_craniofacial_ICH"  ## can be "create_image_list_merge_brain_craniofacial", "create_image_list_merge_brain_craniofacial_ICH    ", "merge_brain_craniofacial", "merge_brain_craniofacial_ICH"


## task 1: create image list for merging brain and craniofacial structures
if task == "create_image_list_merge_brain_craniofacial":  
    brain_structure_rel_paths = get_all_files(brain_structure_segmentation_dir)
    craniofacial_structures_rel_paths = get_all_files(craniofacial_structures_segmentation_dir)
    merged_rel_paths = get_all_files(brain_craniofacial_segmentation_dir)

    common_rel_paths = sorted(set(brain_structure_rel_paths).intersection(set(craniofacial_structures_rel_paths)))
    ## remove already processed ones
    common_rel_paths = [rel_path for rel_path in common_rel_paths if rel_path not in merged_rel_paths]
    
    print(f"Found {len(common_rel_paths)} common IDs.")
    print("First 5 common IDs:")
    print(common_rel_paths[:5])

    ## write common_rel_paths to a text file, this is to be used for batch processing
    with open('/path/to/total_segmentator/brain_craniofacial_to_be_merged_ids.txt', 'w') as f:
        for rel_path in common_rel_paths:
            f.write(f"{rel_path}\n")  

## task 2: merge brain and craniofacial structures
if task == "merge_brain_craniofacial":
    # === Args: id_list_path, start_idx, end_idx ===
    id_list_path = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])

    with open(id_list_path, 'r') as f:
        full_ids = [line.strip() for line in f.readlines()]

    image_paths = full_ids[start_idx:end_idx]


    for rel_path in tqdm(image_paths):
        print(process_image(rel_path))


## task 3: create image list for merging brain and craniofacial structures with cerebral bleed
if task == "create_image_list_merge_brain_craniofacial_ICH":
    # # Collect matching image IDs
    cerebral_bleed_rel_paths = get_all_files(cerebral_bleed_segmentation_dir)
    brain_craniofacial_rel_paths = get_all_files(brain_craniofacial_segmentation_dir)
    merged_rel_paths = get_all_files(brain_craniofacial_ICH_segmentation_dir)

    common_rel_paths = sorted(set(cerebral_bleed_rel_paths).intersection(set(brain_craniofacial_rel_paths)))
    ## remove already processed ones
    common_rel_paths = [rel_path for rel_path in common_rel_paths if rel_path not in merged_rel_paths]

    print(f"Found {len(common_rel_paths)} common IDs.")
    print("First 5 common IDs:")
    print(common_rel_paths[:5])


    # ## write common_rel_paths to a text file, this is to be used for batch processing
    with open('/path/to/total_segmentator/brain_craniofacial_ICH_to_be_merged_ids.txt', 'w') as f:
        for rel_path in common_rel_paths:
            f.write(f"{rel_path}\n")



## task 4: merge brain and craniofacial structures with cerebral bleed
if task == "merge_brain_craniofacial_ICH":

    # === Args: id_list_path, start_idx, end_idx ===
    id_list_path = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])

    with open(id_list_path, 'r') as f:
        full_ids = [line.strip() for line in f.readlines()]

    image_paths = full_ids[start_idx:end_idx]

    for rel_path in tqdm(image_paths):
        print(process_image_add_ich(rel_path))






