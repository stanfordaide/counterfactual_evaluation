import os
import pandas as pd
from data_utils import stratified_train_val_test_split_by_patient
import json
import subprocess
import argparse
import nibabel as nib
from tqdm import tqdm


task = "create_diffusion_model_training_data"  ## run in this order: "create_train_val_test_split", "fill_in_dim_and_spacing" (once for the train and once for the test csv), "create_diffusion_model_training_data", "create_test_data_json"

## Input: the table named in all_info_csv_file_all_non_contrast (below), one row per internal non-contrast head CT,
## with the columns 'Patient Id', 'Age', 'Gender', 'Race', 'ct_nifti_rel_path' (relative to ct_dir), 'Impression' and 'LLM Label'.

ct_dir = "/path/to/head_ct_niis"

output_base_dir = "/path/to/ich_datalists"


if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

embedding_dir = "/path/to/embeddings"


if task == "create_train_val_test_split":
    all_info_csv_file_all_non_contrast = os.path.join(output_base_dir, "ich_starr_head_with_reports_llm_labeled_all_non_contrast_ct_paths.csv")
    with open(all_info_csv_file_all_non_contrast, "r") as f:
        all_info_df_all_non_contrast = pd.read_csv(f)
    print(f"Total number of rows in all_info_df_all_non_contrast: {len(all_info_df_all_non_contrast)}")
    print(f"First 5 rows: {all_info_df_all_non_contrast.head()}")

    ### unique patient ids
    patient_ids = all_info_df_all_non_contrast['Patient Id'].unique().tolist()
    print(f"Total number of unique patient ids: {len(patient_ids)}")  ## 4210

    ## round the age to the nearest integer
    all_info_df_all_non_contrast['Age'] = all_info_df_all_non_contrast['Age'].round(0).astype(int)
    print(f"First 5 rows: {all_info_df_all_non_contrast.head()}")

    ## unique ages
    ages = all_info_df_all_non_contrast['Age'].unique().tolist()
    print(f"Total number of unique ages: {len(ages)}")  ## 101
    print(ages[:5])

    print(min(ages), max(ages))

    ## unique genders
    genders = all_info_df_all_non_contrast['Gender'].unique().tolist()
    print(f"Total number of unique genders: {len(genders)}")  ## 2
    print(genders[:5])

    ## unique races
    races = all_info_df_all_non_contrast['Race'].unique().tolist()
    print(f"Total number of unique races: {len(races)}")  ## 3
    print(races[:5])

    # Stratified train/val/test split by patient_id (0.7 / 0.1 / 0.2), balancing age, gender, race
    train_df, val_df, test_df = stratified_train_val_test_split_by_patient(
        all_info_df_all_non_contrast,
        patient_id_col="Patient Id",
        stratify_cols=["Age", "Gender", "Race"],
        ratios=(0.7, 0.1, 0.2),
        age_col="Age",
        age_bin_edges=None,  # 10-year bins from min to max
        seed=42,
    )
    print(f"Train: {len(train_df)} rows, {train_df['Patient Id'].nunique()} patients")
    print(f"Val:   {len(val_df)} rows, {val_df['Patient Id'].nunique()} patients")
    print(f"Test:  {len(test_df)} rows, {test_df['Patient Id'].nunique()} patients")

    # Save split CSVs
    train_csv = os.path.join(output_base_dir, "train_starr_head_with_reports_llm_labeled.csv")
    val_csv = os.path.join(output_base_dir, "val_starr_head_with_reports_llm_labeled.csv")
    test_csv = os.path.join(output_base_dir, "test_starr_head_with_reports_llm_labeled.csv")
    if os.path.exists(train_csv) or os.path.exists(val_csv) or os.path.exists(test_csv):
        print(f"Train, val or test csv already exists!")
        exit()
    else:
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print(f"Saved train to {train_csv}")
        print(f"Saved val to {val_csv}")
        print(f"Saved test to {test_csv}")

    # Optional: print distribution comparison (age, gender, race) across splits
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        ## print the min and max age
        ## print the min and max age
        print(f"\n{name} - Age mean: {d['Age'].mean():.1f}, Gender: {d['Gender'].value_counts().to_dict()}, Race: {d['Race'].value_counts().to_dict()}")
        print(f"Min age: {d['Age'].min()}, Max age: {d['Age'].max()}")
 


if task == "fill_in_dim_and_spacing":
    # second run: set csv_file to the test table
    csv_file = os.path.join(output_base_dir, "train_starr_head_with_reports_llm_labeled.csv")

    with open(csv_file, "r") as f:
        df = pd.read_csv(f)
    ### add new columns to the df
    df['dim_1'] = None
    df['dim_2'] = None
    df['dim_3'] = None
    df['spacing_1'] = None
    df['spacing_2'] = None
    df['spacing_3'] = None

    for idx, row in tqdm(df.iterrows(), desc="Filling in dim and spacing"):
        image = row['ct_nifti_rel_path']
        ct_path = os.path.join(ct_dir, image)
        
        ct_img = nib.load(ct_path)
        dimensions = ct_img.shape
        dimensions = dimensions[:3]
        spacing = ct_img.header.get_zooms()[:3]
        spacing = spacing[:3]
        spacing = [float(_item) for _item in spacing]
        df.at[idx, 'dim_1'] = dimensions[0]
        df.at[idx, 'dim_2'] = dimensions[1]
        df.at[idx, 'dim_3'] = dimensions[2]
        df.at[idx, 'spacing_1'] = spacing[0]
        df.at[idx, 'spacing_2'] = spacing[1]
        df.at[idx, 'spacing_3'] = spacing[2]

    output_csv_file = csv_file.replace(".csv", "_with_dim_and_spacing.csv")
    df.to_csv(output_csv_file, index=False)
    print(f"Saved df to {output_csv_file}")



if task == "create_diffusion_model_training_data":

    def round_to_tgt_dim_and_spacing(dim, spacing):
         ## head CT has relative fixed spaceing and dim. since spacing and dim will be used as condition for the diffusion model, we need to round the dim and spacing to the nearest multiple of the base number.
        DIM_1 = [512, 512, 512]  
        SPACING_1 = [0.43, 0.43, 0.29]

        DIM_2 = [512, 512, 64]
        SPACING_2 = [0.43, 0.43, 2.9]

        ## if dimensions[2] is more close to 64 than to 512, use DIM_2 and SPACING_2, else use DIM_1 and SPACING_1
        if abs(dim[2] - 64) < abs(dim[2] - 512):
            dim = DIM_2
            spacing = SPACING_2
        else:
            dim = DIM_1
            spacing = SPACING_1
    
        return dim, spacing

    
    train_csv_file = os.path.join(output_base_dir, "train_starr_head_with_reports_llm_labeled_with_dim_and_spacing.csv")
    with open(train_csv_file, "r") as f:
        train_df = pd.read_csv(f)
    print(f"Total number of rows in train_df: {len(train_df)}")
    print(f"First 5 rows: {train_df.head()}")
    ## print the last 10 columns names
    print(f"Last 10 columns names: {train_df.columns[-10:]}")


    train_df['Impression'] = train_df['Impression'].fillna("No impression available.")
    ct_dir = "/path/to/head_ct_niis"
    embedding_dir = "/path/to/embeddings"
    segmentation_dir = "/path/to/total_segmentator/total_segmentator_brain_craniofacial_ICH_merged"
    output_json_file = os.path.join(output_base_dir, "diffusion_train_headct.json")
    data_list = []
    for idx, row in tqdm(train_df.iterrows(), desc="Creating diffusion model training data"):
        item = {}
        image = row['ct_nifti_rel_path']
        embedding_path = os.path.join(embedding_dir, image.replace(".nii", "_emb.nii.gz"))
        segmentation_path = os.path.join(segmentation_dir, image.replace(".nii", "_segmentation_brain_craniofacial_ICH_merged.nii.gz"))
        if os.path.exists(embedding_path) is False or os.path.exists(segmentation_path) is False:
            print(f"Embedding or segmentation file does not exist: {embedding_path} or {segmentation_path}")
            break
        item['image'] = embedding_path
        item['label'] = 1 if row['LLM Label'] == 'positive' else 0

        dim = [int(row['dim_1']), int(row['dim_2']), int(row['dim_3'])]
        spacing = [float(row['spacing_1']), float(row['spacing_2']), float(row['spacing_3'])] ## round it to 2 decimal places
        item['dim'], item['spacing'] = round_to_tgt_dim_and_spacing(dim, spacing)

        item['fold'] = 0
        item['embedding'] = embedding_path
        item['segmentation'] = segmentation_path
        item['ct_image'] = os.path.join(ct_dir, image)
        item['text_prompt'] = row['Impression']
        item['patient_id'] = row['Patient Id']
        item['age'] = int(row['Age'])
        item['gender'] = row['Gender'].lower()
        item['race'] = row['Race'].lower()
        item['text_prompt'] = str(row['Age']) + " years old " + row['Race'].lower() + " " + row['Gender'].lower() + ". IMPRESSION:" + row['Impression']
        data_list.append(item)

    new_datalist = {"training": data_list}
    with open(output_json_file, "w") as f:
        json.dump(new_datalist, f, indent=2)
    print(f"Saved new datalist to {output_json_file} with {len(data_list)} samples")

if task == "create_test_data_json":

    def round_to_tgt_dim_and_spacing(dim, spacing):
         ## head CT has relative fixed spaceing and dim. since spacing and dim will be used as condition for the diffusion model, we need to round the dim and spacing to the nearest multiple of the base number.
        DIM_1 = [512, 512, 512]  
        SPACING_1 = [0.43, 0.43, 0.29]

        DIM_2 = [512, 512, 64]
        SPACING_2 = [0.43, 0.43, 2.9]

        ## if dimensions[2] is more close to 64 than to 512, use DIM_2 and SPACING_2, else use DIM_1 and SPACING_1
        if abs(dim[2] - 64) < abs(dim[2] - 512):
            dim = DIM_2
            spacing = SPACING_2
        else:
            dim = DIM_1
            spacing = SPACING_1
    
        return dim, spacing

    
    test_csv_file = os.path.join(output_base_dir, "test_starr_head_with_reports_llm_labeled_with_dim_and_spacing.csv")
    with open(test_csv_file, "r") as f:
        test_df = pd.read_csv(f)
    print(f"Total number of rows in test_df: {len(test_df)}")
    print(f"First 5 rows: {test_df.head()}")
    ## print the last 10 columns names
    print(f"Last 10 columns names: {test_df.columns[-10:]}")


    test_df['Impression'] = test_df['Impression'].fillna("No impression available.")
    ct_dir = "/path/to/head_ct_niis"
    embedding_dir = "/path/to/embeddings"
    segmentation_dir = "/path/to/total_segmentator/total_segmentator_brain_craniofacial_ICH_merged"
    output_json_file = os.path.join(output_base_dir, "testset_starr_head.json")
    data_list = []
    count_missing = 0
    for idx, row in tqdm(test_df.iterrows(), desc="Creating test data json"):
        item = {}
        image = row['ct_nifti_rel_path']
        embedding_path = os.path.join(embedding_dir, image.replace(".nii", "_emb.nii.gz"))
        segmentation_path = os.path.join(segmentation_dir, image.replace(".nii", "_segmentation_brain_craniofacial_ICH_merged.nii.gz"))
        if os.path.exists(embedding_path) is False or os.path.exists(segmentation_path) is False:
            print(f"Embedding or segmentation file does not exist: {embedding_path} or {segmentation_path}")
            count_missing += 1
            continue
        item['image'] = embedding_path
        item['label'] = 1 if row['LLM Label'] == 'positive' else 0

        dim = [int(row['dim_1']), int(row['dim_2']), int(row['dim_3'])]
        spacing = [float(row['spacing_1']), float(row['spacing_2']), float(row['spacing_3'])] ## round it to 2 decimal places
        item['dim'], item['spacing'] = round_to_tgt_dim_and_spacing(dim, spacing)

        item['fold'] = 0
        item['embedding'] = embedding_path
        item['segmentation'] = segmentation_path
        item['ct_image'] = os.path.join(ct_dir, image)
        # item['text_prompt'] = row['Impression']
        item['patient_id'] = row['Patient Id']
        item['age'] = int(row['Age'])
        item['gender'] = row['Gender'].lower()
        item['race'] = row['Race'].lower()
        item['text_prompt'] = str(row['Age']) + " years old " + row['Race'].lower() + " " + row['Gender'].lower() + ". IMPRESSION:" + row['Impression']
        data_list.append(item)
    ## print the length of data_list
    print(f"Length of data_list: {len(data_list)}")
    ## print the length of test_df
    print(f"Length of test_df: {len(test_df)}")
    print(f"Count of missing samples: {count_missing}")
    new_datalist = {"training": data_list}
    with open(output_json_file, "w") as f:
        json.dump(new_datalist, f, indent=2)
    print(f"Saved new datalist to {output_json_file} with {len(data_list)} samples")
