import os

from transformers import T5Model

# Define the model name and the output path (set the CF_CLINICALT5 env var)
model_name = "luqh/ClinicalT5-base"
output_path = os.environ.get("CF_CLINICALT5", "/path/to/clinicalt5-pytorch")

print(f"Loading original Flax model: {model_name}")

# Load the full T5 model from Flax.
# We load the full model to ensure all weights convert correctly.
model = T5Model.from_pretrained(model_name, from_flax=True)

print(f"Saving converted PyTorch model to: {output_path}")

# Save it in standard PyTorch format
model.save_pretrained(output_path)

print("Conversion complete!")
