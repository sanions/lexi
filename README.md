# Lexi

Lexi is an interactive system that uses computer vision, speech recognition, and generative AI to help users (especially children) learn new words in context. By pointing at words in a book and saying a trigger phrase, Lexi captures the scene, recognizes the word, and provides a kid-friendly definition and a generated image.

---

## Table of Contents

| File/Folder                | Description                                                                                   |
|----------------------------|----------------------------------------------------------------------------------------------|
| `live_trigger.py`          | Main entry point. Runs the live camera, listens for the trigger phrase, and coordinates the hand tracking, OCR, and AI definition/image generation pipeline. |
| `ocr_full.py`              | Contains OCR and pointing logic. Extracts words from images, determines which word is being pointed at, and saves results. |
| `process_target.py`        | Uses Google Gemini API to generate a definition and image for the target word in context. Also includes text-to-speech output. |
| `models/hand_landmarker.task` | Pre-trained model file for MediaPipe hand tracking. Required for hand detection.           |
| `saved_frames/`            | Directory where captured frames, landmarks, and OCR results are saved.                       |
| `images/`                  | Directory where generated images for target words are saved.                                 |
| `uv_lock_to_csv.py`        | Utility script to convert a `uv.lock` file (from the `uv` Python package manager) to a CSV of packages and versions. |
| `pyproject.toml`           | Python project file listing all dependencies and required Python version.                     |
| `uv.lock`                  | Lock file with exact versions of all installed Python packages (used by `uv`).               |
| `all_packages.xlsx`        | (Optional) Excel file, listing all packages (not required for running the code).     |
| `.python-version`          | Specifies the Python version for the project (should match `pyproject.toml`).                |
| `.gitignore`               | Standard gitignore file for Python projects.                                                 |

---

## Setup Instructions

### 1. System Requirements

- **Operating System:** macOS (tested), should work on Linux/Windows with compatible hardware.
- **Python Version:** 3.12 or higher (see `.python-version` and `pyproject.toml`)
- **Hardware:** Webcam and microphone required.

### 2. Install Python

Ensure you have Python 3.12+ installed. You can check with:

```sh
python3 --version
```

If you use `pyenv`, you can install the correct version with:

```sh
pyenv install 3.12.0
pyenv local 3.12.0
```

### 3. Install Dependencies

This project uses the `uv` package manager (a fast drop-in replacement for pip). **Note:** You can install it using its own installer. See [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

- **macOS/Linux (Homebrew):**
  ```sh
  pipx install uv
  ```
- **Or use the official install script:**
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Then, install all dependencies:

```sh
uv sync 
```

Or, if you prefer pip:

```sh
pip install -r requirements.txt
```

**Key dependencies:**
- `mediapipe`
- `opencv-python`
- `pyaudio`
- `pytesseract`
- `pyttsx3`
- `speechrecognition`
- `google-genai`

**System packages:**  
You may need to install system-level dependencies for `pyaudio` and `tesseract-ocr`:

- **macOS:**  
  ```sh
  brew install portaudio tesseract
  ```

- **Ubuntu:**  
  ```sh
  sudo apt-get install portaudio19-dev tesseract-ocr
  ```

### 4. Download/Check Model Files

Ensure the file `models/hand_landmarker.task` exists. This is required for hand tracking. If missing, download from the [official MediaPipe repository](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models).

### 5. Directory Structure

The following directories should exist (the code will create them if needed):

- `saved_frames/triggered/`
- `saved_frames/triggered/ocr/`
- `images/`

### 6. Getting a Gemini API Key

To use the definition and image generation features, you need a Google Gemini API key:

1. Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) and sign in with your Google account.
2. Create a new API key or use an existing one.
3. Set the API key as an environment variable in your shell:
   ```sh
   export GEMINI_API_KEY=your_api_key
   ```
---

### 7. Running the Application

Start the main program:

With uv: 

```sh
uv run live_trigger.py
```

or without uv:

```sh
python live_trigger.py
```

- The webcam will activate.
- Say "Lex" (or the trigger word) to capture a frame and process the pointing gesture.
- The system will save the frame, extract the pointed word, generate a definition and image, and (optionally) speak the definition.

---

## Notes & Troubleshooting

- **Microphone/Camera:** Make sure your system allows Python to access the microphone and camera.
- **API Keys:** The `process_target.py` file uses a Google Gemini API key. Replace the placeholder with your own key if needed.
- **Tesseract Path:** If `pytesseract` cannot find Tesseract, you may need to set the path in your environment or in the code.

---

