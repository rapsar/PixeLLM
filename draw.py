import gradio as gr
import time
import os
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
from fireworks import LLM
from fireworks import client as fw_client

DEFAULT_CANVAS = 64
DEFAULT_BRUSH = 2

def make_blank_canvas(w: int, h: int) -> Image.Image:
    # Grayscale black canvas; ImageEditor will convert to its image_mode
    return Image.new("L", (w, h), 0)

def pil_to_rowstring(img: Image.Image) -> str:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    lines = [",".join(map(str, row.tolist())) + ";" for row in arr]
    return "\n".join(lines)

def pil_to_binstring(img: Image.Image, thresh: int = 128) -> str:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    mask = (arr >= int(thresh)).astype(np.uint8)
    lines = [",".join(map(str, row.tolist())) + ";" for row in mask]
    return "\n".join(lines)

# --- LLM helpers (lazy load per model) ---
_LLM_CACHE = {}  # model_id -> (tokenizer, model)

# --- Fireworks API helper ---
def run_llm_fireworks(prompt: str, max_new_tokens: int, temperature: float, model_id: str) -> str:
    """
    Call Fireworks' Python SDK using the **Completions** API via `LLM`.
    Always uses `llm.completions.create(prompt=...)` for base-model prompt continuation.
    """
    api_key = (os.environ.get("FIREWORKS_API_KEY", "").strip())
    if not api_key:
        return "[Fireworks error: set FIREWORKS_API_KEY in your environment or FIREWORKS_API_KEY_HARDCODED]"

    # Ensure the SDK can see the key if we're using the hardcoded fallback
    if not os.environ.get("FIREWORKS_API_KEY"):
        os.environ["FIREWORKS_API_KEY"] = api_key

    fw_client.api_key = api_key

    try:
        # Use serverless deployment type
        llm = LLM(model=model_id, deployment_type="auto")
        resp = llm.completions.create(
            prompt=prompt,
            max_tokens=int(max_new_tokens),
            temperature=float(temperature) if temperature is not None else 0.0,
        )
        return (getattr(resp.choices[0], "text", "") or "").strip()
    except Exception as e:
        return f"[Fireworks SDK error: {e}]"

def load_llm(model_id: str):
    if model_id in _LLM_CACHE:
        return _LLM_CACHE[model_id]
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    _LLM_CACHE[model_id] = (tok, mdl)
    return tok, mdl

def run_llm(prompt: str, max_new_tokens: int = 64, temperature: float = 0.0, model_id: str = "meta-llama/Llama-3.2-1B") -> str:
    # Fireworks model normalization: allow short, mid, or full ID forms.
    if isinstance(model_id, str) and (
        model_id.startswith("fireworks:") or
        model_id.startswith("fireworks/models/") or
        model_id.startswith("accounts/fireworks/models/")
    ):
        # Normalize to the Fireworks SDK expected form: "accounts/fireworks/models/..."
        if model_id.startswith("fireworks:"):
            fw_model = model_id.split(":", 1)[1]
        elif model_id.startswith("fireworks/models/"):
            fw_model = "accounts/" + model_id
        else:  # already "accounts/fireworks/models/..."
            fw_model = model_id
        # Fireworks serverless models are generally *instruct* tuned.
        # Prepend a brief instruction before the CSV payload so the model knows to continue the sequence.
        instruction = (
            "Continue the sequence below. Don't write anything but the sequence continuation."
        )
        prompt = f"{instruction}\n\n{prompt}"
        return run_llm_fireworks(prompt, int(max_new_tokens), float(temperature), fw_model)

    tok, mdl = load_llm(model_id)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(next(mdl.parameters()).device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = mdl.generate(
            inputs["input_ids"],
            max_new_tokens=int(max_new_tokens),
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text.strip()

def csv_single_line(csv_multiline: str) -> str:
    # Remove newlines; keep semicolons as row delimiters
    return (csv_multiline or "").replace("\n", "")

def parse_csv_image(s: str, width: int):
    # Parse a semicolon/comma separated string of integers into an L-mode image
    try:
        rows = [r for r in s.strip().split(";") if r != ""]
        parsed_rows = []
        for r in rows:
            nums = []
            for tok in r.split(","):
                tok = ''.join(ch for ch in tok if ch.isdigit())
                if tok == "":
                    continue
                v = max(0, min(255, int(tok)))
                nums.append(v)
            if nums:
                # pad/truncate to the canvas width
                if len(nums) < width:
                    nums = nums + [0] * (width - len(nums))
                else:
                    nums = nums[:width]
                parsed_rows.append(nums)
        if not parsed_rows:
            return None
        arr = np.array(parsed_rows, dtype=np.uint8)
        return Image.fromarray(arr, mode="L")
    except Exception:
        return None

def apply_settings(canvas_px):
    w = int(canvas_px)
    h = int(canvas_px)
    # Recreate the editor with consistent config and a fresh blank canvas to enforce size
    return gr.ImageEditor(
        canvas_size=(w, h),
        value=make_blank_canvas(w, h),
        image_mode="RGBA",
        brush=gr.Brush(
            default_size=DEFAULT_BRUSH,
            colors=["black", "#404040", "#808080", "#C0C0C0", "white"],
            default_color="white",  # white stands out on the new black canvas
            color_mode="fixed",
        ),
        eraser=gr.Eraser(default_size=1),
        transforms=("crop", "resize"),
        height=500,
    )
    
def sleep(im):
    time.sleep(5)
    return [im["background"], im["layers"][0], im["layers"][1], im["composite"]]

# Process uploaded image: resize to canvas width, grayscale, update editor + preview
def process_upload(im, canvas_px, scale, invert, binarize, bin_thresh):
    if not im or im.get("background") is None:
        return None, None
    bg = im["background"]
    img = Image.fromarray(bg)
    # convert to grayscale
    img = img.convert("L")
    # resize to canvas width, keep aspect
    w, h = img.size
    target_w = int(canvas_px) if canvas_px is not None else w
    if target_w <= 0:
        target_w = w
    target_h = max(1, round(h * target_w / max(1, w)))
    resized = img.resize((target_w, target_h), Image.LANCZOS)

    # Create a canvas-sized grayscale image and paste the resized image at (0,0)
    canvas_gray = Image.new("L", (target_w, target_w), 0)
    canvas_gray.paste(resized, (0, 0))

    # Editor value (canvas-size, grayscale)
    editor_value = canvas_gray

    # Preview & CSV: start from canvas_gray, optionally invert, then
    # - CSV from canvas-sized image
    # - Preview from upscaled image
    base_for_text = canvas_gray
    if invert:
        base_for_text = ImageOps.invert(base_for_text)
    if bool(binarize):
        text = pil_to_binstring(base_for_text, bin_thresh)
    else:
        text = pil_to_rowstring(base_for_text)

    s = max(1, int(scale) if scale is not None else 8)
    preview = base_for_text.resize((base_for_text.width * s, base_for_text.height * s), Image.NEAREST)
    return editor_value, preview, text

def make_preview(im, scale, invert, binarize, bin_thresh):
    if im is None or im.get("composite") is None:
        return None, ""
    arr = im["composite"]
    base = Image.fromarray(arr).convert("L")  # canvas-sized grayscale
    # Apply inversion for both preview and CSV (CSV stays canvas-sized)
    base_for_text = ImageOps.invert(base) if invert else base
    if bool(binarize):
        text = pil_to_binstring(base_for_text, bin_thresh)
    else:
        text = pil_to_rowstring(base_for_text)

    # Preview is the upscaled version of base_for_text
    s = max(1, int(scale) if scale is not None else 8)
    preview = base_for_text.resize((base_for_text.width * s, base_for_text.height * s), Image.NEAREST)
    return preview, text

def extrapolate_with_llm(csv_text, canvas_px, out_rows, model_id):
    one_line = csv_single_line(csv_text)
    # Count how many rows come from the input (non-empty segments ending with ';')
    input_rows_count = len([r for r in (one_line or "").split(";") if r.strip()])
    try:
        width = int(canvas_px)
    except Exception:
        width = DEFAULT_CANVAS
    max_tokens = int(out_rows) * width * 2
    prompt = one_line  # feed the single-line CSV directly
    try:
        gen = run_llm(prompt, int(max_tokens), model_id=model_id)
    except Exception as e:
        gen = f"[LLM error: {e}]"
        return gen, None

    # Parse INPUT + OUTPUT together; ';' marks end-of-row
    combined = (one_line or "") + (gen or "")
    rows = [r for r in combined.split(";") if r.strip()]

    parsed = []
    max_w = 0
    for r in rows:
        vals = []
        for tok in r.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = int(float(tok))
            except Exception:
                continue
            # clamp to 0-255 grayscale
            if v < 0: v = 0
            if v > 255: v = 255
            vals.append(v)
        if vals:
            parsed.append(vals)
            if len(vals) > max_w:
                max_w = len(vals)

    if not parsed:
        return gen, None

    # Pad rows to the full width so we can render the full rectangular image
    arr_rows = []
    for vals in parsed:
        if len(vals) < max_w:
            vals = vals + [0] * (max_w - len(vals))
        else:
            vals = vals[:max_w]
        arr_rows.append(vals)

    import numpy as np
    arr = np.array(arr_rows, dtype=np.uint8)
    # If the array is binary (only 0 and 1), rescale to 0-255
    if set(np.unique(arr).tolist()).issubset({0, 1}):
        arr = arr * 255
    img = Image.fromarray(arr, mode="L")

    # Resize to width=512, preserve aspect ratio
    target_w = 512
    orig_w, orig_h = img.size
    target_h = max(1, round(orig_h * target_w / max(1, orig_w)))
    img = img.resize((target_w, target_h), Image.NEAREST)

    # Draw a thin red separator line at the boundary between input and output rows
    # Map input row index from original height to resized height
    if input_rows_count > 0 and orig_h > 0:
        y = round(input_rows_count * target_h / orig_h)
        y = max(0, min(target_h - 1, y))
        img_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)
        draw.line([(0, y), (img_rgb.width - 1, y)], fill=(255, 0, 0), width=1)
        img = img_rgb

    display_text = (gen or "").replace(";", ";\n")
    return display_text, img

# themes
theme = gr.Theme.from_hub('gstaff/xkcd')
theme.set(block_background_fill="#7ffacd8e",)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("### Extrapolate images with LLMs")

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            canvas_px = gr.Slider(32, 128, value=DEFAULT_CANVAS, step=1, label="Canvas size (px)")
            # brush_px = gr.Slider(1, 16, value=DEFAULT_BRUSH, step=1, label="Brush size (px)")
            preview_scale = gr.Slider(1, 16, value=8, step=1, label="Preview scale (×)")
            invert_preview = gr.Checkbox(value=False, label="Invert preview")

            with gr.Accordion("Binarize", open=False):
                binarize_csv = gr.Checkbox(value=False, label="Turn 0-255 into 0/1")
                bin_thresh = gr.Slider(0, 255, value=128, step=1, label="Threshold")

            out_rows_default_value = 3
            out_rows = gr.Slider(1, 16, value=out_rows_default_value, step=1, label="Number of output rows")
            llm_choice = gr.Dropdown(
                label="LLM model",
                choices=[
                    "meta-llama/Llama-3.2-1B",
                    "meta-llama/Llama-3.2-3B",
                    "meta-llama/Llama-3.1-8B",
                    "HuggingFaceTB/SmolLM2-1.7B",
                    "HuggingFaceTB/SmolLM3-3B-Base",
                    # Fireworks serverless (base models via /completions)
                    # ("fireworks:accounts/fireworks/models/llama-v3p2-1b", "fireworks/models/llama-v3p2-1b"), # not serverless
                    # ("fireworks:accounts/fireworks/models/llama-v3p2-3b", "fireworks/models/llama-v3p2-3b"), # not serverless
                    # These are all instruct models, so need to add a quick instruction to the prompt
                    # All these models have tokenizers with 0 to 999 as single tokens
                    "fireworks/models/gpt-oss-20b",
                    "fireworks/models/gpt-oss-120b",
                    "fireworks/models/llama-v3p1-8b-instruct",
                    "fireworks/models/llama-v3p1-70b-instruct",
                    "fireworks/models/llama-v3p1-405b-instruct",
                    "fireworks/models/llama-v3p3-70b-instruct",
                ],
                value="meta-llama/Llama-3.2-1B",
            )
            api_help = gr.Markdown("Set <code>FIREWORKS_API_KEY</code> in your environment to use the Fireworks options.")
            out_tokens_info = gr.Markdown(f"**Output tokens:** {DEFAULT_CANVAS * out_rows_default_value * 2}")

        with gr.Row(scale=4):
            im = gr.ImageEditor(
                type="numpy",
                canvas_size=(DEFAULT_CANVAS, DEFAULT_CANVAS),
                image_mode="RGBA",
                brush=gr.Brush(
                    default_size=DEFAULT_BRUSH,
                    colors=["black", "#404040", "#808080", "#C0C0C0", "white"],
                    default_color="black",
                    color_mode="fixed",
                ),
                eraser=gr.Eraser(default_size=1),
                transforms=("crop", "resize"),
                # fixed_canvas=True,
                height=500,
            )
            im_preview = gr.Image(height=512, label="Preview (scaled)")
    preview_text = gr.Textbox(label="Preview as CSV (rows end with ';')", lines=12, interactive=False, show_copy_button=True, max_lines=5)
    # Helper to update button label
    def update_button_label(model_id):
        return f"Extrapolate with LLM ({model_id})"

    extrap_btn = gr.Button(value=update_button_label(llm_choice.value if hasattr(llm_choice, "value") else "meta-llama/Llama-3.2-1B"))
    llm_text = gr.Textbox(label="LLM output (single-line CSV)", lines=6, interactive=False, show_copy_button=True)
    llm_image = gr.Image(label="LLM parsed image", height=512)

    # Update the editor when either control changes
    canvas_px.change(apply_settings, inputs=[canvas_px], outputs=im)
    canvas_px.change(make_preview, inputs=[im, preview_scale, invert_preview, binarize_csv, bin_thresh], outputs=[im_preview, preview_text])
    # brush_px.change(apply_settings, inputs=[canvas_px, brush_px], outputs=im)
    
    im.upload(process_upload, inputs=[im, canvas_px, preview_scale, invert_preview, binarize_csv, bin_thresh], outputs=[im, im_preview, preview_text])
    im.change(make_preview, inputs=[im, preview_scale, invert_preview, binarize_csv, bin_thresh], outputs=[im_preview, preview_text], show_progress="hidden")
    preview_scale.change(make_preview, inputs=[im, preview_scale, invert_preview, binarize_csv, bin_thresh], outputs=[im_preview, preview_text])
    invert_preview.change(make_preview, inputs=[im, preview_scale, invert_preview, binarize_csv, bin_thresh], outputs=[im_preview, preview_text])
    binarize_csv.change(make_preview, inputs=[im, preview_scale, invert_preview, binarize_csv, bin_thresh], outputs=[im_preview, preview_text])
    bin_thresh.change(make_preview, inputs=[im, preview_scale, invert_preview, binarize_csv, bin_thresh], outputs=[im_preview, preview_text])
    extrap_btn.click(extrapolate_with_llm, inputs=[preview_text, canvas_px, out_rows, llm_choice], outputs=[llm_text, llm_image])

    # Update button label dynamically when LLM model changes
    llm_choice.change(update_button_label, inputs=[llm_choice], outputs=[extrap_btn])

    def update_tokens(out_rows, canvas_px):
        try:
            width = int(canvas_px)
        except Exception:
            width = DEFAULT_CANVAS
        tokens = int(out_rows) * width * 2
        return f"**Output tokens:** {tokens}"

    out_rows.change(update_tokens, inputs=[out_rows, canvas_px], outputs=out_tokens_info)
    canvas_px.change(update_tokens, inputs=[out_rows, canvas_px], outputs=out_tokens_info)

    demo.load(update_tokens, inputs=[out_rows, canvas_px], outputs=out_tokens_info)

if __name__ == "__main__":
    demo.launch()
