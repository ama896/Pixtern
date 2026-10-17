"""
Pixtern ✨  —  a photo magic studio
Built with Streamlit (web app) + OpenCV & Pillow (image processing) + rembg (AI background removal)

Run it with:   streamlit run app.py
"""

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

st.set_page_config(page_title="Pixtern", page_icon="✨", layout="wide")

MAX_SIDE = 1600  # big photos are shrunk to this size so the app stays fast


# =====================================================================
# 1. EFFECTS — each takes an RGB image (NumPy array) and returns one
# =====================================================================

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gray_to_rgb(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def pencil_sketch(img, blur=21):
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(255 - gray, (blur, blur), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return gray_to_rgb(sketch)


def color_sketch(img, blur=21):
    shading = pencil_sketch(img, blur).astype(np.float32) / 255
    return np.clip(img * shading, 0, 255).astype(np.uint8)


def cartoon(img, smoothness=3):
    color = img.copy()
    for _ in range(smoothness):  # flatten colours but keep edges sharp
        color = cv2.bilateralFilter(color, 9, 75, 75)
    edges = cv2.adaptiveThreshold(  # bold black outlines
        cv2.medianBlur(to_gray(img), 7), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    return cv2.bitwise_and(color, color, mask=edges)


def sepia(img):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    return np.clip(img.astype(np.float32) @ kernel.T, 0, 255).astype(np.uint8)


def vignette(img, amount=0.5):
    h, w = img.shape[:2]
    sigma = 1.2 - amount  # bigger amount = darker corners
    mask = cv2.getGaussianKernel(h, h * sigma) @ cv2.getGaussianKernel(w, w * sigma).T
    mask = mask / mask.max()
    return (img * mask[..., None]).astype(np.uint8)


def pixel_art(img, block=12, colors=6):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_AREA)
    step = 256 // colors  # fewer colours = more retro look
    small = np.clip((small.astype(np.int32) // step) * step + step // 2, 0, 255).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def edge_detection(img, low=50, high=150):
    edges = cv2.Canny(to_gray(img), low, high)
    return gray_to_rgb(255 - edges)  # black lines on white paper


def emboss(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.filter2D(img, -1, kernel)


def blur(img, amount=10):
    k = amount * 2 + 1  # kernel size must be an odd number
    return cv2.GaussianBlur(img, (k, k), 0)


EFFECTS = ["Original", "Grayscale", "Pencil Sketch", "Color Sketch", "Cartoon",
           "Vintage", "Pixel Art", "Edge Detection", "Emboss", "Blur", "Invert"]


def apply_effect(img, name, opts):
    if name == "Grayscale":
        return gray_to_rgb(to_gray(img))
    if name == "Pencil Sketch":
        return pencil_sketch(img, opts["blur"])
    if name == "Color Sketch":
        return color_sketch(img, opts["blur"])
    if name == "Cartoon":
        return cartoon(img, opts["smoothness"])
    if name == "Vintage":
        return vignette(sepia(img), opts["amount"])
    if name == "Pixel Art":
        return pixel_art(img, opts["block"], opts["colors"])
    if name == "Edge Detection":
        return edge_detection(img, opts["low"], opts["high"])
    if name == "Emboss":
        return emboss(img)
    if name == "Blur":
        return blur(img, opts["amount"])
    if name == "Invert":
        return 255 - img
    return img.copy()  # "Original"


# =====================================================================
# 2. AI TOOLS — background removal (rembg) and face detection (OpenCV)
# =====================================================================

@st.cache_resource(show_spinner="Loading the AI background model (first time only, ~170 MB)...")
def load_rembg_session():
    from rembg import new_session
    return new_session("u2net")


@st.cache_data(show_spinner="Finding the subject...")
def foreground_mask(img):
    """Returns a 0-255 mask: 255 where the person/object is, 0 for background."""
    from rembg import remove
    cutout = remove(Image.fromarray(img), session=load_rembg_session())
    return np.array(cutout)[:, :, 3]


@st.cache_resource
def load_face_detector():
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(path)
        return None if detector.empty() else detector
    except Exception:
        return None


def draw_faces(img, source):
    detector = load_face_detector()
    if detector is None:
        return img, None
    faces = detector.detectMultiScale(to_gray(source), scaleFactor=1.1,
                                      minNeighbors=5, minSize=(40, 40))
    out = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 120), 3)
    return out, len(faces)


# =====================================================================
# 3. SIDEBAR — all the controls
# =====================================================================

st.sidebar.title("✨ Pixtern")
uploaded = st.sidebar.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

st.sidebar.header("🎨 Effect")
effect = st.sidebar.selectbox("Choose an effect", EFFECTS)
opts = {}
if effect in ("Pencil Sketch", "Color Sketch"):
    opts["blur"] = st.sidebar.slider("Sketch softness", 3, 51, 21, step=2)
elif effect == "Cartoon":
    opts["smoothness"] = st.sidebar.slider("Colour smoothness", 1, 5, 3)
elif effect == "Vintage":
    opts["amount"] = st.sidebar.slider("Dark corners", 0.1, 1.0, 0.5)
elif effect == "Pixel Art":
    opts["block"] = st.sidebar.slider("Pixel size", 2, 40, 12)
    opts["colors"] = st.sidebar.slider("Colours per channel", 2, 16, 6)
elif effect == "Edge Detection":
    opts["low"], opts["high"] = st.sidebar.slider("Edge sensitivity", 0, 300, (50, 150))
elif effect == "Blur":
    opts["amount"] = st.sidebar.slider("Blur amount", 1, 30, 10)

with st.sidebar.expander("🎛️ Adjustments"):
    brightness = st.slider("Brightness", 0.2, 2.0, 1.0, 0.05)
    contrast = st.slider("Contrast", 0.2, 2.0, 1.0, 0.05)
    saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
    rotate = st.selectbox("Rotate (degrees)", [0, 90, 180, 270])
    mirror = st.checkbox("Mirror (left ↔ right)")
    flip = st.checkbox("Flip (top ↕ bottom)")

st.sidebar.header("🤖 AI tools")
bg_mode = st.sidebar.selectbox(
    "Background", ["Keep", "Remove (transparent)", "Solid colour", "Blur (portrait mode)"])
bg_color = st.sidebar.color_picker("Background colour", "#4F8BF9") if bg_mode == "Solid colour" else None
detect = st.sidebar.checkbox("Detect faces")


# =====================================================================
# 4. MAIN PAGE — process the photo and show the results
# =====================================================================

st.title("Pixtern ✨")
st.caption("Upload a photo, pick an effect, and download your creation.")

if not uploaded:
    st.info("👈 Upload a photo from the sidebar to get started.")
    st.stop()

# Load the photo (exif_transpose fixes sideways phone photos)
photo = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
photo.thumbnail((MAX_SIDE, MAX_SIDE))

# Step 1: adjustments
adjusted = ImageEnhance.Brightness(photo).enhance(brightness)
adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast)
adjusted = ImageEnhance.Color(adjusted).enhance(saturation)
adjusted = ImageEnhance.Sharpness(adjusted).enhance(sharpness)
if rotate:
    adjusted = adjusted.rotate(-rotate, expand=True)
if mirror:
    adjusted = ImageOps.mirror(adjusted)
if flip:
    adjusted = ImageOps.flip(adjusted)
base = np.array(adjusted)

# Step 2: effect
result = apply_effect(base, effect, opts)

# Step 3: face detection
face_count = None
if detect:
    result, face_count = draw_faces(result, base)
    if face_count is None:
        st.warning("Face detector couldn't load in this OpenCV version, so faces were skipped.")

# Step 4: background
if bg_mode != "Keep":
    mask = foreground_mask(base)
    if bg_mode.startswith("Remove"):
        result = np.dstack([result, mask])  # add transparency channel
    else:
        alpha = (mask.astype(np.float32) / 255)[..., None]
        if bg_mode == "Solid colour":
            rgb = tuple(int(bg_color[i:i + 2], 16) for i in (1, 3, 5))
            background = np.full_like(result, rgb)
        else:
            background = cv2.GaussianBlur(result, (0, 0), 15)
        result = (result * alpha + background * (1 - alpha)).astype(np.uint8)

# Show before and after
left, right = st.columns(2)
with left:
    st.subheader("Before")
    st.image(photo)
with right:
    st.subheader("After")
    st.image(result)

# Quick stats
m1, m2, m3 = st.columns(3)
m1.metric("Size", f"{base.shape[1]} × {base.shape[0]} px")
m2.metric("Effect", effect)
m3.metric("Faces found", face_count if face_count is not None else "—")

# Download button
buffer = io.BytesIO()
Image.fromarray(result).save(buffer, format="PNG")
st.download_button(
    "⬇️ Download result (PNG)", buffer.getvalue(),
    file_name=f"Pixtern_{effect.lower().replace(' ', '_')}.png", mime="image/png")

# Colour histogram
with st.expander("📊 Colour histogram"):
    rgb_only = result[:, :, :3]
    hist = pd.DataFrame({
        name: np.bincount(rgb_only[:, :, i].ravel(), minlength=256)
        for i, name in enumerate(["Red", "Green", "Blue"])
    })
    st.line_chart(hist, color=["#FF4B4B", "#3DD56D", "#4F8BF9"])