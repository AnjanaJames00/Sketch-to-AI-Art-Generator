import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Caption model
# -------------------------
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# -------------------------
# CLIP embeddings
# -------------------------
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# -------------------------
# ControlNet
# -------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

pipe.enable_attention_slicing()

# -------------------------
# Feedback dataset
# -------------------------
embeddings = []
labels = []

# -------------------------
# Caption from sketch
# -------------------------
def caption_image(image):

    inputs = blip_processor(
        image,
        return_tensors="pt"
    ).to(device)

    out = blip_model.generate(**inputs)

    caption = blip_processor.decode(
        out[0],
        skip_special_tokens=True
    )

    # remove words that bias art styles
    caption = caption.replace("drawing of", "")
    caption = caption.replace("sketch of", "")
    caption = caption.replace("a drawing", "")

    return caption.strip()


# -------------------------
# Prompt builder
# -------------------------
def build_prompt(caption, style):

    if style == "photorealistic":
        style_prompt = "ultra realistic photo, DSLR, natural lighting, sharp focus"

    elif style == "anime":
        style_prompt = "anime style, vibrant colors"

    elif style == "studio ghibli":
        style_prompt = "studio ghibli style, soft lighting, whimsical"

    elif style == "oil painting":
        style_prompt = "oil painting on canvas, thick brush strokes"

    elif style == "pencil sketch":
        style_prompt = "detailed pencil sketch"

    else:
        style_prompt = style

    return f"{caption}, {style_prompt}, highly detailed"


# -------------------------
# Generate image
# -------------------------
def generate(image, style):

    if image is None:
        return None, "No image"

    if isinstance(image, dict):
        image = image.get("composite", None)

    img_np = np.array(image)

    caption = caption_image(image)

    # Edge detection
    edge = cv2.Canny(img_np, 100, 200)
    edge = np.stack([edge]*3, axis=2)
    edge = Image.fromarray(edge).resize((512,512))

    prompt = build_prompt(caption, style)

    negative_prompt = "painting, oil painting, illustration, cartoon, blurry, low quality"

    result = pipe(
        prompt=prompt,
        image=edge,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=1.5
    ).images[0]

    return result, caption


# -------------------------
# Save feedback
# -------------------------
def save_feedback(image, correct_label):

    try:

        if image is None:
            return "No image"

        if isinstance(image, dict):
            image = image.get("composite", None)

        inputs = clip_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)

        embeddings.append(emb.cpu().numpy())
        labels.append(correct_label)

        return f"Saved examples: {len(labels)}"

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("# Sketch → AI Art")

    canvas = gr.ImageEditor(type="pil")

    style = gr.Dropdown(
        [
            "photorealistic",
            "anime",
            "studio ghibli",
            "oil painting",
            "pencil sketch"
        ],
        value="photorealistic"
    )

    btn = gr.Button("Generate")

    output = gr.Image()
    caption_box = gr.Textbox(label="AI Description")

    correct_label = gr.Textbox(label="Correction")
    feedback_btn = gr.Button("Save Feedback")
    status = gr.Textbox()

    btn.click(
        generate,
        inputs=[canvas, style],
        outputs=[output, caption_box]
    )

    feedback_btn.click(
        save_feedback,
        inputs=[canvas, correct_label],
        outputs=status
    )

demo.launch()