import sys
import torch
import base64
import urllib.parse
from torch import autocast
from diffusers import StableDiffusionPipeline

from flask import Flask, request
app = Flask(__name__)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
pipe = pipe.to(device)

output_file_name = "output.png"
sentence = sys.argv[1]
print("Generating magic image from sentence: " + sentence)

with autocast("cuda"):
  image = pipe(sentence, guidance_scale=7.5)["sample"][0]  
image.save(output_file_name)
