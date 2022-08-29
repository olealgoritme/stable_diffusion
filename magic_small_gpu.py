import sys
import torch
import base64
import urllib.parse
from torch import autocast
from diffusers import StableDiffusionPipeline

from flask import Flask, request
app = Flask(__name__)

def image_to_data_url(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    with open(filename, 'rb') as f:
        img = f.read()
    return prefix + base64.b64encode(img).decode('utf-8')

# http://server:5000/magic?create=cute little monkey sitting on a beach enjoying the sun
@app.route('/magic', methods=['GET'])
def magic_image_route():
  encoded_string = request.args.get("create")
  sentence = urllib.parse.unquote(encoded_string)
  print("Incoming Sentence: " + sentence)
  magic_image = get_magic_from_sentence(sentence)
  base64_img = image_to_data_url(magic_image)
  return '''<img src=\"{}\"/>'''.format(base64_img)

def get_magic_from_sentence(sentence):
  model_id = "CompVis/stable-diffusion-v1-4"
  device = "cuda"

  pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
  pipe = pipe.to(device)

  output_file_name = "output.png"
  print("Generating magic image from sentence: " + sentence)

  with autocast("cuda"):
      image = pipe(sentence, guidance_scale=7.5)["sample"][0]  
  image.save(output_file_name)
  return output_file_name
