# https://huggingface.co/docs/transformers/en/model_doc/kosmos-2
import random
from PIL import Image, ImageDraw
import numpy as np
import requests
from transformers import AutoProcessor, Kosmos2Model, Kosmos2ForConditionalGeneration
import torch
import os
import utils

TEACHER_FORCING = int(os.environ.get('TEACHER_FORCING', 0))
GENERATION = int(os.environ.get('GENERATION', 0))

device = utils.get_device()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = None
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image = Image.open(requests.get(url, stream=True).raw)

if TEACHER_FORCING:
    ##### TEACHER FORCING -> TRAINING #####

    model = Kosmos2Model.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)
    # (il mio sito)[http://mysite.com]
    text = (
        "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863>"
        "</object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911>"
        "</object>"
    )

    inputs = processor(text=text, images=image, return_tensors="pt", add_eos_token=True).to(device)

    inv_vocab = {v: k for k, v in processor.tokenizer.vocab.items()}
    # inv_vocab[64924]

    outputs = model(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
    )

if GENERATION:
    ##### GENERATION -> INFERENCE #####

    if model:
        del model
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)

    prompt = "<grounding> An image of"
    # prompt = "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object>"

    gen_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        pixel_values=gen_inputs["pixel_values"],
        input_ids=gen_inputs["input_ids"],
        attention_mask=gen_inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=gen_inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=64,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    print(processed_text)

    caption, entities = processor.post_process_generation(generated_text)
    print(caption)

    print(entities)

    draw = ImageDraw.Draw(image)
    size = image.size[0]
    colors = list(utils.COLORS)
    random.shuffle(colors)
    for (_, __, bb_p), color in zip(entities, colors):
        bb = np.round(np.array(bb_p[0]) * size)
        draw.rectangle(((bb[0], bb[1]), (bb[2], bb[3])), outline=color, width=5)
    grounded_image = draw._image
    ...