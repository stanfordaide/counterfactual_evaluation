import re
import subprocess
import json
import os
import csv
from datetime import datetime
from tqdm import tqdm
import torch
import multiprocessing as mp
from queue import Empty
import time
import argparse


def extract_seed_from_filename(filename: str) -> int:
    """
    Extract the seed number from the filename.
    
    Args:
        filename (str): The filename to extract the seed from.
        
    Returns:
        int: The extracted seed number, or 0 if not found.
    """
    # Look for '_seed' followed by any number of digits
    match = re.search(r'_seed(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no seed found 

def run_single_inference(args):
    """
    Run inference for a single case on a specific GPU.
    
    Args:
        args (tuple): Tuple containing (test_case, gpu_id, idx, demo)
        
    Returns:
        tuple: (image_id, impression, label, prediction_filename, seed, gpu_id) or None if failed
    """
    test_case, gpu_id, idx, demo = args
    
    impression = test_case["text_prompt"]
    image_id = test_case.get("image_id", f"case_{idx}")
    label = test_case.get("label", "unknown")
    
    if demo:
        age = test_case["age"]
        race = test_case["race"]
        sex = test_case["gender"] 
        impression = f"{str(age)} year old {race} {sex}. {impression}"
        env_config = "configs/environment_maisi_diff_model_text_demo.json"
    else:
        env_config = "configs/environment_maisi_diff_model_text.json"
    # Set CUDA_VISIBLE_DEVICES for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print(f"impression -> {impression}")
    # Run inference with single GPU
    cmd = [
            "torchrun",
            "--nproc_per_node=1",  # Use single GPU
            f"--master_port={12355 + gpu_id}",  # Different port for each GPU
            "scripts/diff_model_infer_text.py",
            "--env_config", env_config,
            "--model_config", "configs/config_maisi_diff_model.json",
            "--model_def", "configs/config_maisi3d-rflow.json",
            "--num_gpus", "1",  # Use single GPU
            "--prompts", impression,
            "--gpu_id", str(gpu_id)  # Pass GPU ID to inference script
        ]
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        
        # Extract the output filename from the command output
        output_lines = result.stdout.split('\n')
        prediction_filename = None
        
        for i, line in enumerate(output_lines):
            if "Generated output saved to:" in line:
                prediction_filename = line.split("Generated output saved to:")[1].strip()
                break
            elif "Saved" in line and ".nii.gz" in line:
                prediction_filename = line.split("Saved")[1].strip().rstrip('.')
                break
        
        if prediction_filename is None:
            return None
        
        # Extract seed from filename
        seed = extract_seed_from_filename(prediction_filename)
        
        print(f"Completed inference for case {idx} on GPU {gpu_id}")
        if demo:
            return (image_id, impression, label, prediction_filename, seed, gpu_id, age, race, sex)
        else:
            return (image_id, impression, label, prediction_filename, seed, gpu_id)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running inference for case {idx} on GPU {gpu_id}:")
        print(f"Command failed with exit code {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        return None

def csv_writer_process(csv_path, csv_headers, queue, demo):
    """
    Process to handle CSV writing from a queue.
    
    Args:
        csv_path (str): Path to the CSV file
        csv_headers (list): List of CSV column headers
        queue (mp.Queue): Queue to receive results
    """
    print(f"Starting CSV writer process, writing to {csv_path}")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        csvfile.flush()  # Ensure header is written immediately
        
        while True:
            try:
                result = queue.get(timeout=1)  # Wait 1 second for new results
                if result is None:  # Poison pill to stop the process
                    print("CSV writer received poison pill, stopping...")
                    break
                print(f'demo -> {demo}')    
                if demo:
                    image_id, impression, label, prediction_filename, seed, gpu_id, age, race, sex = result
                    row = {
                        'image_id': image_id,
                        'impression': impression,
                        'label': label,
                        'prediction_filename': prediction_filename,
                        'seed': seed,
                        'gpu_id': gpu_id,
                        'age': age,
                        'race': race,
                        'sex': sex
                    }
                else:
                    image_id, impression, label, prediction_filename, seed, gpu_id = result
                    row = {
                        'image_id': image_id,
                        'impression': impression,
                        'label': label,
                        'prediction_filename': prediction_filename,
                        'seed': seed,
                        'gpu_id': gpu_id
                    }
                writer.writerow(row)
                csvfile.flush()  # Ensure data is written immediately
            except Empty:
                continue  # No results yet, keep waiting
            except Exception as e:
                print(f"Error writing to CSV: {e}")
                continue

def run_batch_inference(num_gpus: int = 2, demo: bool = False):
    """
    Run batch inference using distributed training.
    
    Args:
        num_gpus (int): Number of GPUs to use for distributed inference.
    """
    if num_gpus > torch.cuda.device_count():
        raise RuntimeError(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available!")
    
    print(f"Running inference on {num_gpus} GPUs")
    
    if demo:
        # Load the test data list
        with open("/path/to/inspect2/pe_training_data_all/pe_datalist_spacing_all_test_with_demo.json", "r") as f:
            test_data = json.load(f)
        
        # Load environment config to get model name
        with open("configs/environment_maisi_diff_model_text_demo.json", "r") as f:
            env_config = json.load(f)
        model_name = env_config.get("output_prefix", "unet_3d_text_all_l2_loss_best_demo")
        # Create output directory if it doesn't exist
        output_dir = "/path/to/inspect2/pe_training_data_all/predictions/unet_3d_text_all_l2_loss_best_demo"
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Load the test data list
        with open("/path/to/inspect2/pe_training_data_all/pe_datalist_spacing_all_test.json", "r") as f:
            test_data = json.load(f)
        
        # Load environment config to get model name
        with open("configs/environment_maisi_diff_model_text.json", "r") as f:
            env_config = json.load(f)
        model_name = env_config.get("output_prefix", "unet_3d_text_all_l2_loss_best")
        # Create output directory if it doesn't exist
        output_dir = "/path/to/inspect2/pe_training_data_all/predictions/unet_3d_text_all_l2_loss_best"
        os.makedirs(output_dir, exist_ok=True)
    
    print(model_name)
    # Create CSV file to track predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"inference_results_{model_name}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Prepare CSV headers
    if demo:
        csv_headers = ['image_id', 'impression', 'label', 'prediction_filename', 'seed', 'gpu_id', 'age', 'race', 'sex']
    else:
        csv_headers = ['image_id', 'impression', 'label', 'prediction_filename', 'seed', 'gpu_id']
    
    # Create a queue for results
    result_queue = mp.Queue()
    
    # Start CSV writer process
    writer_process = mp.Process(
        target=csv_writer_process,
        args=(csv_path, csv_headers, result_queue, demo)
    )
    writer_process.start()
    
    # Remove existing predictions from test data if they exist
    # load csv file, get image_if column
    import pandas as pd
    
    existing_image_ids = None
    if demo:
        print('no existing predictions for demo')
        # csv_filename_existing_preds = "/path/to/inspect2/pe_training_data_all/inference_results_unet_3d_text_all_l2_loss_best_demo_20250730_160000.csv"
    else:
        csv_filename_existing_preds = "/path/to/inspect2/pe_training_data_all/inference_results_unet_3d_text_all_l2_loss_20250630_205622.csv"
        df = pd.read_csv(csv_filename_existing_preds)
        existing_image_ids = df["image_id"].tolist()
    
    if existing_image_ids is not None:
        #remove existing image ids from test data
        print(f'before: {len(test_data["test"])}')
        test_data["test"] = [case for case in test_data["test"] if case["image_id"] not in existing_image_ids]
        print(f'after: {len(test_data["test"])}')
    
    try:
        # Prepare arguments for parallel processing
        args_list = []
        for idx, test_case in enumerate(test_data["test"]):
            gpu_id = idx % num_gpus
            args_list.append((test_case, gpu_id, idx, demo))
            
        
        # Create a pool of workers
        with mp.Pool(num_gpus) as pool:
            # Process in batches of 8
            batch_size = num_gpus
            for i in range(0, len(args_list), batch_size):
                batch = args_list[i:i + batch_size]
                # Run inference in parallel with progress bar
                for result in tqdm(
                    pool.imap_unordered(run_single_inference, batch),
                    total=len(batch),
                    desc=f"Running inference batch {i//batch_size + 1}"
                ):
                    if result is not None:
                        result_queue.put(result)
                
                # Add a delay between batches
                if i + batch_size < len(args_list):
                    print(f"Waiting 2 seconds before starting next batch...")
                    time.sleep(2)
        
        print("All inference tasks completed, sending poison pill to CSV writer...")
        # Send poison pill to stop writer process
        result_queue.put(None)
        writer_process.join()
        
    except Exception as e:
        print(f"Error during inference: {e}")
        result_queue.put(None)  # Stop writer process
        writer_process.join()
        raise
    
    print(f"\nInference results saved to: {csv_path}")


parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", type=int, default=6)
# Use text conditioning with demographics or not
parser.add_argument("--demo", action='store_true', default=False, help="Use demo mode with demographics")

args = parser.parse_args()

if __name__ == "__main__":
    run_batch_inference(num_gpus=args.num_gpus, demo=args.demo) 
