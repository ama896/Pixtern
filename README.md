# Pixtern ✨

**Give your photos some magic.** Pixtern is a photo-editing web app built with Python. Upload any photo, turn it into a sketch, cartoon or pixel art, remove or blur the background with AI, detect faces, and download the result, all in your browser.

> My first Python + image processing project. 🚀

---

## Features

### 🎨 11 artistic effects
| Effect | What it does |
|---|---|
| Grayscale | Classic black and white |
| Pencil Sketch | Turns the photo into a hand-drawn pencil sketch |
| Color Sketch | A sketch that keeps the original colours |
| Cartoon | Smooth colours with bold black outlines |
| Vintage | Sepia tones with dark, faded corners |
| Pixel Art | Retro 8-bit look with adjustable pixel size and colours |
| Edge Detection | Finds and draws the outlines in the photo |
| Emboss | A raised, 3D metal-stamp look |
| Blur | Soft Gaussian blur |
| Invert | Photo-negative colours |
| Original | No effect |

Most effects come with their own slider to fine-tune the result.

### 🎛️ Photo adjustments
Brightness, contrast, saturation and sharpness sliders, plus rotate, mirror and flip.

### 🤖 AI tools
- **Background removal**: cut out the subject and save it as a transparent PNG
- **Background replace**: put the subject on any solid colour
- **Portrait mode**: blur the background like a phone camera
- **Face detection**: draws boxes around faces and counts them

### ✅ Extras
- Side-by-side **before / after** view
- **Download** the result as a PNG
- **Colour histogram** of the edited image
- Automatically fixes sideways phone photos and resizes huge images so the app stays fast

---

## Built with

- **[Python 3.13](https://www.python.org/)**
- **[Streamlit](https://streamlit.io/)**: turns the Python script into a web app
- **[OpenCV](https://opencv.org/)**: image effects and face detection
- **[Pillow](https://python-pillow.org/)** and **[NumPy](https://numpy.org/)**: loading and processing images
- **[rembg](https://github.com/danielgatis/rembg)**: AI background removal (U²-Net model)
- **[pandas](https://pandas.pydata.org/)**: colour histogram data

---

## Run it on your computer

**1. Clone the project**
```bash
git clone https://github.com/ama896/Pixtern.git
cd Pixtern
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

**3. Install the libraries**
```bash
pip install -r requirements.txt
```

**4. Start the app**
```bash
streamlit run app.py
```
Your browser opens the app at `http://localhost:8501`.

> **Note:** the first time you use background removal, the AI model (~170 MB) downloads automatically, so give it a minute.

---

## How it works

Every photo goes through four steps:

1. **Adjust**: brightness, contrast, saturation, sharpness, rotation and flips (Pillow)
2. **Effect**: the chosen artistic filter (OpenCV and NumPy)
3. **Faces**: optional face detection with a Haar cascade classifier (OpenCV)
4. **Background**: optional AI cut-out, colour fill or blur (rembg)

---

## Ideas for the future

- [ ] More effects (oil painting, watercolour, neon glow)
- [ ] Before / after comparison slider
- [ ] Batch editing of several photos at once
- [ ] Deploy online with Streamlit Community Cloud

---

## Author

**Vaman S** · [@ama896](https://github.com/ama896)

If you like this project, give it a ⭐!