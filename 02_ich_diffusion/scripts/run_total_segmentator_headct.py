import json
import os
import subprocess
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import argparse
import shutil

def download_model_weights(output_dir):
    """Download TotalSegmentator model weights if not present"""
    # Set the model weights directory
    model_dir = output_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    # Download model weights
    os.environ['TOTALSEG_HOME_DIR'] = output_dir
    os.makedirs(model_dir, exist_ok=True)
    # Run totalseg_download_weights command
    # cmd = ["totalseg_download_weights", "-t", "total"]
    cmd = ["totalseg_download_weights", "-t", "cerebral_bleed"]
    # cmd = ["totalseg_download_weights", "-t", "brain_structures"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Model weights downloaded successfully to {model_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model weights: {e.stderr}")
        raise

def process_single_case(args):
    """Process a single case with TotalSegmentator"""
    ct_path, output_dir, gpu_id, model_dir, task= args
    
    # Create output directory for this case
    case_name = os.path.basename(ct_path).split('.')[0]
    case_output_dir = os.path.join(output_dir, f"{case_name}_segmentation_{task}.nii.gz")

   #case_output_dir = os.path.join(output_dir, f"{case_name}_segmentation_{args.task}.nii.gz")
    
    print(f"Model directory: {model_dir}")
    
    # Set the model weights directory for this process
    os.environ['TOTALSEG_HOME_DIR'] = model_dir
    
    # Run TotalSegmentator with specific GPU
    cmd = [
        "TotalSegmentator",
        "-i", ct_path,
        "-o", case_output_dir,
        "--ml",
        "--device", f"gpu:{gpu_id}",
        "--task", task,
    ]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, case_name
    except subprocess.CalledProcessError as e:
        print(f"Error processing {case_name} on GPU {gpu_id}: {e.stderr}")
        return False, case_name

def run_total_segmentator_parallel(args):
    """Run TotalSegmentator in parallel on multiple GPUs"""
    # Set up model weights directory
    model_dir = args.model_dir
    print(f"Model directory: {model_dir}")

    

    # Download model weights if needed
    if not os.path.exists(os.path.join(model_dir, "nnunet/results")):
        print("Model weights not found. Downloading...")
        try:
            download_model_weights(model_dir)
        except Exception as e:
            print("Failed to download model weights. Please check your internet connection and try again.")
            return
    
    # Read dataset.json
    with open(args.dataset_json_path, 'r') as f:
        dataset = json.load(f)
    
    # Create output directory
    output_dir = os.path.join(args.output_base_dir, "total_segmentator_"+args.task)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    # base_dir = os.path.dirname(os.path.dirname(dataset_json_path))  # Go up one level to get to inspect2
    args_list = []
    for i, item in enumerate(dataset['training']):
        # Construct path with CTPA directory
        ct_path = os.path.join(args.ct_dir, item['image'])
        if not os.path.exists(ct_path):
            print(f"Warning: File not found: {ct_path}")
            continue
        #if file already in total_segmentator_output_dir:
        case_name = os.path.basename(ct_path).split('.')[0]
        case_output_dir = os.path.join(output_dir, os.path.dirname(item['image']))
        if os.path.exists(os.path.join(case_output_dir, f"{case_name}_segmentation_{args.task}.nii.gz")):
            print(f"Skipping {case_name} because it already exists in {case_output_dir}")
            continue
        # Assign GPU ID in round-robin fashion
        gpu_id = i % args.num_gpus
        args_list.append((ct_path, case_output_dir, gpu_id, model_dir, args.task))
    
    print(f"Total files to process: {len(args_list)}")
    print(args_list[0])  # Print first 5 for verification
    
    if not args_list:
        print("No valid files found to process!")
        return
        
    # Set up multiprocessing
    num_processes = min(args.num_gpus, len(args_list))
    print(f"Running TotalSegmentator on {num_processes} GPUs")
    print(f"Found {len(args_list)} files to process")
    
    # Process cases in parallel
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_case, args_list),
            total=len(args_list),
            desc="Processing cases"
        ))
    
    # Print summary
    successful = sum(1 for success, _ in results if success)
    failed = sum(1 for success, _ in results if not success)
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed cases:")
        for success, case_name in results:
            if not success:
                print(f"- {case_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TotalSegmentator in parallel on multiple GPUs")

    parser.add_argument(
        "--ct_dir",
        type=str,
        default="/path/to/head_ct_niis",
        help="Directory containing CT scans",
    )

    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default = "/path/to/dataset.json",
        help="Path to dataset.json file"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/path/to/total_segmentator",
        help="Base directory for outputs"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="/path/to/total_segmentator",
        help="Directory for model weights"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="cerebral_bleed",
        help="task for total segmentator (e.g., brain_structures, cerebral_bleed, craniofacial_structures)"
    )


    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--ct_path",
        type=str,
        default="/path/to/head_ct_niis/<STUDY_ID>.nii",
        help="Path to a single CT file to process"  
    )


    
    args = parser.parse_args()
    
    # process_single_case((args.ct_path, args.output_base_dir, 0, args.model_dir, args.task))

    if os.path.exists(args.output_base_dir) is False:
        os.makedirs(args.output_base_dir, exist_ok=True)
    
    run_total_segmentator_parallel(args)


    
    # run_total_segmentator_parallel(
    #     args.dataset_json,
    #     args.output_dir,
    #     args.num_gpus
    # ) 
