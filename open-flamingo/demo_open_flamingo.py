# https://github.com/mlfoundations/open_flamingo?tab=readme-ov-file
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch
from open_flamingo import create_model_and_transforms
import utils
import os

device = utils.get_device()
IN_CONTEXT_LEARNING = int(os.environ.get('IN_CONTEXT_LEARNING', 1))
CAPTIONING = int(os.environ.get('CAPTIONING', 0))

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to(device)

from PIL import Image
import requests
import torch

"""
Step 1: Load images
"""
if IN_CONTEXT_LEARNING:
    in_context_image_one = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw if CAPTIONING else \
        "open-flamingo/assets/latte-art.jpg"
        # "open-flamingo/assets/underground.jpg"
    )

    in_context_image_two = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            stream=True
        ).raw if CAPTIONING else \
        "open-flamingo/assets/among-us.png"
        # "open-flamingo/assets/hard-rock.jpg"
    )

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
        stream=True
    ).raw if CAPTIONING else \
    "open-flamingo/assets/tesla.jpg"
    # "open-flamingo/assets/apple-ipod.jpg"
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
if IN_CONTEXT_LEARNING:
    vision_x = [image_processor(in_context_image_one).unsqueeze(0), image_processor(in_context_image_two).unsqueeze(0)]
else:
    vision_x = []

vision_x.append(image_processor(query_image).unsqueeze(0))
vision_x = torch.cat(vision_x, dim=0).to(device)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
if IN_CONTEXT_LEARNING:
    if CAPTIONING:
        lang_str = "<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|>"
    else:
        lang_str ="<image>Question: What latte art is presented in the image? Answer: A swan.<|endofchunk|><image>Question: Which video game is represented in the image? Answer: Among Us.<|endofchunk|>"
        # lang_str ="<image>Output: \"UNDERGROUND\"<|endofchunk|><image>Output: \"Hard Rock Cafe Beijing\"<|endofchunk|>"
else:
    lang_str = ""

lang_str += ("<image>An image of" if CAPTIONING else "<image>Question: What vehicle is featured in the image? Answer:")
# lang_str += "<image>Output:"
# lang_str += ("<image>Describe the image briefly:" if CAPTIONING else "<image>Question: What vehicle is featured in the image? Answer:")
lang_x = tokenizer([lang_str], return_tensors="pt", ).to(device)
print(f"Input: {lang_str}")

"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

decoded_text = tokenizer.decode(generated_text[0])
print("Generated text: ", decoded_text)
...