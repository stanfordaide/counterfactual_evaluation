import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.utils import instantiate

class TextControlNet(ControlNetMaisi):
    def __init__(self, *args, **kwargs):
        
        # This argument is not defined in contronet_maisi that's why 
        # we need to pop it before instantiating the class
        text_proj_def = kwargs.pop("text_projection_def", None)

        super().__init__(*args, **kwargs)
        # Currently not used, the embeddings are passed as context with dimension 512
        if text_proj_def is not None:
            self.text_proj = instantiate(text_proj_def, __mode="default")
        else:
            self.text_proj = nn.Identity()

    def forward(self, x, timesteps, **kwargs):
        # Get the model's dtype from an existing parameter
        model_dtype = next(self.parameters()).dtype
        
        # Apply the embedding to controlnet_cond if it exists
        if "controlnet_cond" in kwargs:
            # Ensure controlnet_cond is in the same dtype as the model
            kwargs["controlnet_cond"] = kwargs["controlnet_cond"].to(dtype=model_dtype)
        
        # Ensure all tensor inputs have the correct dtype
        if "context" in kwargs and torch.is_tensor(kwargs["context"]):
            kwargs["context"] = kwargs["context"].to(dtype=model_dtype)
        
        return super().forward(x, timesteps, **kwargs) 