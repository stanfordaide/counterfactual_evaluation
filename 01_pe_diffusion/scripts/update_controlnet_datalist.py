"""
Update datalist JSON:
  - Replace segmentation paths to point to total_segmentator_all_lung_pe/
  - Only include entries where the new segmentation file exists
"""

import json
import os

_PE_DATA = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "pe_training_data_all")
INPUT_JSON = os.path.join(_PE_DATA, 'pe_datalist_spacing_all_val_with_demo_3_controlnet.json')
OUTPUT_JSON = os.path.join(_PE_DATA, 'pe_datalist_spacing_all_val_with_demo_5_controlnet.json')
NEW_SEG_DIR = os.path.join(_PE_DATA, 'total_segmentator_all_lung_pe')

with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

# Build set of available segmentation files
available = set(os.listdir(NEW_SEG_DIR))

original_count = 0
kept_count = 0

for split in data:
    if not isinstance(data[split], list):
        continue

    filtered = []
    for entry in data[split]:
        original_count += 1
        image_id = entry.get('image_id', '')
        new_seg_filename = f"{image_id}_segmentation_merged.nii.gz"

        if new_seg_filename not in available:
            continue

        entry['segmentation'] = os.path.join(NEW_SEG_DIR, new_seg_filename)
        filtered.append(entry)
        kept_count += 1

    data[split] = filtered

with open(OUTPUT_JSON, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Original entries: {original_count}")
print(f"Kept entries:     {kept_count}")
print(f"Dropped:          {original_count - kept_count}")
print(f"Saved to: {OUTPUT_JSON}")