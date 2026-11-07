"""Streamlit app for both models: fine-tuned Stable Diffusion (text to image, image to text, image to image)
and the model trained from scratch on Fashion-MNIST (text to image)."""
import io
import math
import os
import threading
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).parent
TEST_DIR = ROOT / "test"
BASE_MODEL = os.environ.get("BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
LORA_FILE = Path(os.environ.get("LORA_FILE", ROOT / "weights" / "pytorch_lora_weights.safetensors"))
SCRATCH_FILE = Path(os.environ.get("SCRATCH_FILE", ROOT / "weights" / "diffusion_from_scratch.pt"))

RESOLUTION = 512
HAS_GPU = torch.cuda.is_available()
DEVICE, DTYPE = ("cuda", torch.float16) if HAS_GPU else ("cpu", torch.float32)

FINE_TUNED, FROM_SCRATCH = "Fine-tuned Stable Diffusion", "From scratch (Fashion-MNIST)"
TEXT_TO_IMAGE, IMAGE_TO_TEXT, IMAGE_TO_IMAGE = "Text to image", "Image to text", "Image to image"
WRITE_OWN = "Write my own..."
UPLOAD_OWN = "Upload my own..."


@st.cache_resource(show_spinner="Loading Stable Diffusion (the first run downloads about 4 GB)...")
def load_diffusion(lora_file):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, safety_checker=None, requires_safety_checker=False, dtype=DTYPE, use_safetensors=True,
    ).to(DEVICE)
    # Gives cleaner images than the default scheduler at low step counts, which matters most on a CPU.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
    )
    if lora_file:
        path = Path(lora_file)
        pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name="finetuned")
    pipe.set_progress_bar_config(disable=True)
    # dtype must be repeated: from_pipe would otherwise convert the shared model to float32.
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(pipe, dtype=DTYPE)
    img2img_pipe.set_progress_bar_config(disable=True)
    return pipe, img2img_pipe


@st.cache_resource(show_spinner="Loading BLIP (the first run downloads about 1 GB)...")
def load_captioner():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    captioner = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL, dtype=DTYPE, use_safetensors=True).to(DEVICE)
    return processor, captioner


@st.cache_resource(show_spinner="Loading the from-scratch model and CLIP (the first run downloads about 600 MB)...")
def load_scratch_model(checkpoint_file):
    from diffusion_from_scratch.scratch_model import ScratchModel

    return ScratchModel(checkpoint_file, DEVICE)


@st.cache_resource(show_spinner=False)
def generation_lock():
    # Tabs share one model, so one image is drawn at a time: another tab could otherwise switch the LoRA mid-image.
    return threading.Lock()


def read_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def choose_text(label, sample_file, key):
    # A sample from the test folder, or the user's own text.
    choice = st.selectbox(label, read_lines(sample_file) + [WRITE_OWN], key=f"{key}_text_choice")
    if choice == WRITE_OWN:
        return st.text_area("Your text", key=f"{key}_text_own").strip()
    return choice


def choose_image(key):
    # A sample from the test folder, or the user's own upload. Returns None until an image is available.
    samples = {path.stem.replace("_", " ").capitalize(): path for path in sorted((TEST_DIR / "images").glob("*.png"))}
    choice = st.selectbox("Image: choose a sample or upload your own", list(samples) + [UPLOAD_OWN],
                          key=f"{key}_image_choice")
    source = samples.get(choice)
    if choice == UPLOAD_OWN:
        source = st.file_uploader("Your image", type=["png", "jpg", "jpeg", "webp"], key=f"{key}_image_upload")
        if source is None:
            return None
    try:
        image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        st.error("That file could not be read as an image: it may be damaged or too large.")
        return None
    st.image(image, caption="Input image", width=320)
    return image


def generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def set_lora(pipe, has_lora, use_lora):
    if not has_lora:
        return
    if use_lora:
        pipe.enable_lora()
    else:
        pipe.disable_lora()


def text2img(prompt, use_lora, has_lora, steps, guidance, seed, on_step):
    pipe, _ = load_diffusion(str(LORA_FILE) if has_lora else None)
    set_lora(pipe, has_lora, use_lora)
    return pipe(prompt, height=RESOLUTION, width=RESOLUTION, num_inference_steps=steps, guidance_scale=guidance,
                generator=generator(seed), callback_on_step_end=on_step).images[0]


def img2img(image, prompt, strength, use_lora, has_lora, steps, guidance, seed, on_step):
    pipe, img2img_pipe = load_diffusion(str(LORA_FILE) if has_lora else None)
    set_lora(pipe, has_lora, use_lora)
    # Cropping to a square keeps the proportions of an uploaded image.
    image = ImageOps.fit(image, (RESOLUTION, RESOLUTION))
    return img2img_pipe(prompt, image=image, strength=strength, num_inference_steps=steps, guidance_scale=guidance,
                        generator=generator(seed), callback_on_step_end=on_step).images[0]


def img2text(image, max_new_tokens=30):
    processor, captioner = load_captioner()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.inference_mode():
        output = captioner.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True)


def model_variants(has_lora, compare):
    # (label, use_lora) for each image to draw.
    if not has_lora:
        return [("Base model", False)]
    return [("Fine-tuned", True), ("Base model", False)] if compare else [("Fine-tuned", True)]


def draw_images(mode, make_image, variants):
    # Draws each variant with a progress bar, because an image can take minutes on a CPU.
    results = []
    for label, use_lora in variants:
        bar = st.progress(0.0, text=f"{label}: starting...")

        def on_step(pipe, step, timestep, callback_kwargs):
            bar.progress((step + 1) / pipe.num_timesteps, text=f"{label}: step {step + 1} of {pipe.num_timesteps}")
            return callback_kwargs

        with generation_lock():
            results.append((label, to_png(make_image(use_lora, on_step))))
        bar.empty()
    st.session_state[f"results_{mode}"] = results


def scratch_grid(images, columns=4, cell=112, gap=4):
    # Enlarges each 28x28 image without smoothing, so its pixels stay sharp, and joins them into one picture.
    rows = math.ceil(len(images) / columns)
    grid = Image.new("L", (columns * cell + (columns - 1) * gap, rows * cell + (rows - 1) * gap), 255)
    for i, image in enumerate(images):
        pixels = ((image[0].float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
        tile = Image.fromarray(pixels).resize((cell, cell), Image.NEAREST)
        grid.paste(tile, ((i % columns) * (cell + gap), (i // columns) * (cell + gap)))
    return grid


def show_images(mode):
    results = st.session_state.get(f"results_{mode}", [])
    for column, (label, png) in zip(st.columns(len(results)) if results else [], results):
        column.image(png, caption=label, width="stretch")
        column.download_button(f"Download: {label}", png, file_name=f"{label.lower().replace(' ', '_')}.png",
                               mime="image/png", key=f"download_{mode}_{label}")


def weights_path_text(path):
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def from_scratch_page():
    has_scratch = SCRATCH_FILE.exists()
    with st.sidebar:
        st.header("⚙️ Settings")
        steps = st.slider("Sampling steps", 10, 100, 50)
        guidance = st.slider("Guidance scale", 0.0, 10.0, 3.0, step=0.5,
                             help="0 ignores the prompt. Around 2-4 usually works best.")
        seed = st.number_input("Seed", min_value=0, value=1337, step=1)
        count = st.slider("Number of images", 1, 16, 8)
        st.divider()
        st.caption(f"Device: {torch.cuda.get_device_name(0) if HAS_GPU else 'CPU'}")
        st.caption(f"From-scratch weights: {'loaded' if has_scratch else 'not found'}")

    st.caption("This model was trained from random weights on 28×28 greyscale clothing images, so it can only draw "
               "the 10 Fashion-MNIST categories, such as a jacket, a sneaker or a bag.")
    if not has_scratch:
        st.warning(f"From-scratch weights not found at `{weights_path_text(SCRATCH_FILE)}`. "
                   "See the README for how to add them.")
    prompt = choose_text("Prompt: choose a sample or write your own", TEST_DIR / "from_scratch_prompts.txt", "scratch")
    if st.button("Generate images", type="primary", disabled=not prompt or not has_scratch):
        model = load_scratch_model(str(SCRATCH_FILE))
        category, score, known = model.closest_category(prompt)
        png = None
        if known:
            bar = st.progress(0.0, text="Starting...")

            def on_step(i, total):
                bar.progress((i + 1) / total, text=f"Step {i + 1} of {total}")

            with generation_lock():
                images = model.generate(prompt, count, guidance, steps, seed, on_step)
            bar.empty()
            png = to_png(scratch_grid(images))
        st.session_state["scratch_result"] = (prompt, category, score, png)

    if "scratch_result" in st.session_state:
        prompt, category, score, png = st.session_state["scratch_result"]
        if png is None:
            st.warning(f"“{prompt}” is not one of the 10 categories this model knows (closest: {category}, "
                       f"similarity {score:.3f}), so nothing was drawn.")
        else:
            st.caption(f"Closest category: {category} (similarity {score:.3f})")
            st.image(png, caption=prompt, width="stretch")
            st.download_button("Download images", png, file_name="from_scratch.png", mime="image/png",
                               key="download_scratch")


st.set_page_config(page_title="Diffusion Models", page_icon="🎨")
st.title("🎨 Diffusion Models: Image Generation and Captioning")

with st.sidebar:
    model_choice = st.radio("Model", [FINE_TUNED, FROM_SCRATCH],
                            help="Stable Diffusion fine-tuned on flower photos, or a small model trained from scratch "
                                 "on Fashion-MNIST.")

if model_choice == FROM_SCRATCH:
    from_scratch_page()
    st.stop()

has_lora = LORA_FILE.exists()

with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Inference steps", 10, 50, 25 if HAS_GPU else 10)
    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5, step=0.5)
    seed = st.number_input("Seed", min_value=0, value=1337, step=1)
    compare = st.checkbox("Also draw with the base model", value=HAS_GPU, disabled=not has_lora,
                          help="Draws a second image from the same seed, which doubles the time.")
    st.divider()
    st.caption(f"Device: {torch.cuda.get_device_name(0) if HAS_GPU else 'CPU'}")
    st.caption(f"Fine-tuned weights: {'loaded' if has_lora else 'not found'}")

if not has_lora:
    st.warning(f"Fine-tuned weights not found at `{weights_path_text(LORA_FILE)}`, "
               "so images use the base model only. See the README for how to add them.")
if not HAS_GPU:
    st.info("No GPU found. On a CPU, an image can take several minutes: fewer inference steps make it faster. "
            "Captions take only seconds.")

mode = st.radio("What would you like to do?", [TEXT_TO_IMAGE, IMAGE_TO_TEXT, IMAGE_TO_IMAGE], horizontal=True)
st.divider()

if mode == TEXT_TO_IMAGE:
    prompt = choose_text("Prompt: choose a sample or write your own", TEST_DIR / "text2img_prompts.txt", "text2img")
    if st.button("Generate image", type="primary", disabled=not prompt):
        draw_images(mode, lambda use_lora, on_step: text2img(prompt, use_lora, has_lora, steps, guidance, seed, on_step),
                    model_variants(has_lora, compare))
    show_images(mode)

elif mode == IMAGE_TO_TEXT:
    st.caption("Captions come from BLIP, a separate model that is not fine-tuned here.")
    image = choose_image("img2text")
    if st.button("Describe image", type="primary", disabled=image is None):
        with st.spinner("Describing..."):
            st.session_state["caption"] = img2text(image)
    if "caption" in st.session_state:
        st.subheader("Caption")
        st.write(st.session_state["caption"])

else:
    image = choose_image("img2img")
    prompt = choose_text("Prompt: choose a sample or write your own", TEST_DIR / "img2img_prompts.txt", "img2img")
    strength = st.slider("Strength: low stays close to the input, high follows the prompt", 0.1, 1.0, 0.6, step=0.05)
    if st.button("Change image", type="primary", disabled=image is None or not prompt):
        draw_images(mode, lambda use_lora, on_step: img2img(image, prompt, strength, use_lora, has_lora, steps, guidance,
                                                            seed, on_step),
                    model_variants(has_lora, compare))
    show_images(mode)
