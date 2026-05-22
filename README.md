# Multimodal Document Analyzer

An AI-powered Streamlit app for analyzing PDFs, images, scanned documents, and text files.

## Features

- PDF, image, scan, text, markdown, and CSV upload
- Native PDF text extraction with OCR fallback for scanned pages
- Image OCR using Tesseract
- Table extraction from PDFs and structured text
- Chart and handwriting-like visual hints using OpenCV
- Extractive summaries, key insights, classification, entity extraction, topic detection
- Cross-document intelligent search
- Question answering over uploaded documents with evidence passages
- Similar document grouping

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On this machine, you can also start the app with:

```bat
start_app.bat
```

Or with PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_app.ps1
```

Then open:

```text
http://127.0.0.1:8503
```

For OCR, install the Tesseract desktop binary as well:

- Windows: install from the UB Mannheim Tesseract builds, then ensure `tesseract.exe` is on PATH.
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

The app still works without Tesseract, but scanned documents and handwritten text extraction will be limited.

## Notes

This implementation uses local NLP and computer vision so it can run without an API key. The question answering and summaries are evidence-based extractive methods over TF-IDF retrieval. You can later add an LLM provider for generative answers while keeping the same extraction pipeline.
