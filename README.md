# Sarkari-Sarathi — AI Digital Scribe for Local Government (Nepal)

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)
![Whisper](https://img.shields.io/badge/Whisper-Nepali%20Fine--tuned-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN%20Classifier-red.svg)

**एक AI-संचालित डिजिटल स्क्राइब जसले नेपाली नागरिकहरूलाई सरकारी कागजातहरू सजिलै उत्पन्न गर्न मद्दत गर्दछ।**

*An AI-powered digital scribe helping Nepali citizens easily generate government documents.*

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [API](#-api-endpoints) • [Contributing](#-contributing)

</div>

---

## Overview

**Sarkari-Sarathi** is a comprehensive document generation system designed for Nepal's local government services. It provides three input methods — voice, handwriting, and keyboard — so that citizens of all literacy levels can fill out official government forms easily.

All AI components run **locally on CPU** — no GPU, no paid API, fully offline-capable.

### What It Does
1. User selects a document type (birth registration, death registration, etc.)
2. Fills in the form using **voice** (Nepali speech recognition), **handwriting** (canvas + CNN classifier), or **keyboard** (with English-to-Nepali transliteration)
3. The system generates a **print-ready PDF** in official government format

## 🎯 Features

### Multi-Modal Input
| Input Method | Description | Technology |
|---|---|---|
| 🎤 **आवाज (Voice)** | Speak in Nepali, get Devanagari text | Fine-tuned Nepali Whisper ASR + Whisper tiny (CPU fallback) |
| ✍️ **हस्तलेखन (Handwriting)** | Draw/write on canvas, get recognized text | CNN word classifier (PyTorch) + Tesseract.js OCR (fallback) |
| ⌨️ **किबोर्ड (Keyboard)** | Type in English, auto-transliterate to Nepali | Custom rule-based transliteration with 60+ conjunct patterns |

### Supported Government Documents (9 Templates)

| Category | Documents |
|---|---|
| **Civil Registration** | जन्म दर्ता (Birth) · मृत्यु दर्ता (Death) · विवाह दर्ता (Marriage) · सम्बन्धविच्छेद (Divorce) |
| **Certificates** | बसाइसराई प्रमाणपत्र (Migration) · बसोबास प्रमाणपत्र (Residence) |
| **Utilities** | विद्युत जडान (Electricity) · खानेपानी जडान (Water) · बाटो पहुँच (Road Access) |

### Key Highlights
- **Fully Free & Offline** — No paid APIs, no GPU required, runs entirely on CPU
- **No Login Required** — Stateless, session-based operation
- **Elder-Friendly UI** — Simple 3-step guided flow (Select → Fill → Download)
- **Cascading Location Dropdowns** — All 7 provinces, 77 districts, 700+ municipalities
- **Real-time Transliteration** — Type English, see Nepali instantly
- **Grammar Correction** — Rule-based Nepali particle and punctuation normalization
- **Bilingual Fields** — Supports both Nepali and English input where needed
- **Alternatives Picker** — CNN returns top-k word choices with confidence scores

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** + **Uvicorn** | REST API server |
| **Python 3.11** | Core runtime |
| **HuggingFace Transformers** | ASR model inference |
| **Fine-tuned Nepali Whisper** | Primary speech recognition (`amitpant7/Nepali-Automatic-Speech-Recognition`) |
| **OpenAI Whisper (tiny)** | Fallback speech recognition (CPU-optimized, 72MB) |
| **PyTorch CNN Classifier** | Handwriting word recognition (237 Nepali word classes, ~140K params) |
| **ReportLab** | PDF generation with Nepali font support |
| **PyDub + FFmpeg** | Audio format conversion (via `imageio-ffmpeg`) |
| **Rule-based Grammar** | Nepali text correction (particle attachment, दण्ड punctuation) |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5 / Tailwind CSS** | Responsive UI |
| **Vanilla JavaScript** | Form logic, transliteration engine |
| **Tesseract.js** | Client-side OCR fallback for handwriting (Nepali + English) |
| **Canvas API** | Free-form handwriting input |
| **MediaRecorder API** | Voice recording from browser |

### AI Models (All Local, All Free)
| Model | Role |
|---|---|
| `amitpant7/Nepali-Automatic-Speech-Recognition` | Primary Nepali ASR (fine-tuned Whisper) |
| `openai/whisper-tiny` | Fallback ASR (CPU-optimized) |
| `NepaliWordCNN` | Handwriting word classifier (237 classes, PyTorch) |
| `Tesseract.js` | Client-side OCR fallback |

## 🏗️ Architecture

```
┌───────────────────────────┐     ┌───────────────────────────┐     ┌────────────────────┐
│        Frontend           │     │         Backend           │     │    Local Models    │
│       (Browser)           │◄───►│        (FastAPI)          │◄───►│   (CPU only)       │
│                           │     │                           │     │                    │
│  • Voice Recording        │     │  • /transcribe-audio      │     │  Nepali Whisper    │
│  • Canvas Handwriting     │     │  • /recognize-handwriting │     │  Whisper (tiny)    │
│  • English→Nepali Translit│     │  • /generate-document     │     │  CNN Word Classif. │
│  • Tesseract.js OCR       │     │  • /transliterate         │     │                    │
│  • Alternatives Picker    │     │  • /correct-grammar       │     └────────────────────┘
│  • Cascading Dropdowns    │     │  • /locations             │
│                           │     │  • PDF Generation         │
│                           │     │  • Grammar Correction     │
│                           │     │  • FFmpeg Audio Convert   │
└───────────────────────────┘     └───────────────────────────┘
```

### Processing Pipelines

**Voice Pipeline:**
```
Mic → MediaRecorder (WebM) → /transcribe-audio → FFmpeg (→WAV 16kHz) → Nepali Whisper → Grammar Correction → Field
```

**Handwriting Pipeline:**
```
Canvas Drawing → /recognize-handwriting → CNN Word Classifier (237 classes)
                                              ↓ top-k alternatives
                                         Alternatives Picker → User selects → Grammar Correction → Field
                                              ↓ (low confidence fallback)
                                         Tesseract.js OCR (client-side)
```

**Keyboard Pipeline:**
```
English Keystrokes → Real-time Transliteration (60+ conjunct rules) → Nepali Devanagari → Field
```

## 📁 Project Structure

```
Sarkari-Sarathi/
├── backend/
│   ├── main.py                  # FastAPI app — all API endpoints, PDF generation
│   ├── nepali_asr.py            # Nepali ASR module (Whisper fine-tuned + FFmpeg setup)
│   ├── grammar.py               # Rule-based Nepali grammar correction
│   ├── locations.json           # Nepal administrative data (7 provinces, 77 districts, 700+ municipalities)
│   ├── requirements.txt         # Python dependencies
│   ├── templates/               # 9 document templates (JSON)
│   │   ├── birth_registration.json
│   │   ├── death_registration.json
│   │   ├── marriage_registration.json
│   │   ├── divorce_registration.json
│   │   ├── migration_certificate.json
│   │   ├── residence_certificate.json
│   │   ├── electricity_connection.json
│   │   ├── water_connection.json
│   │   └── road_access.json
│   ├── generated_documents/     # Output PDFs (auto-created)
│   └── static/
│       ├── fonts/               # NotoSansDevanagari font for PDF
│       └── handwriting_model/   # CNN model checkpoint + metadata
│           ├── nepali_word_cnn.pt
│           ├── model_meta.json
│           └── vocab.json
├── frontend/
│   ├── index.html               # Main application UI (with alternatives picker)
│   ├── script.js                # Core logic — transliteration, forms, voice, dropdowns, CNN UI
│   └── tesseract_handwriting.js # Tesseract.js OCR wrapper with preprocessing
├── handwriting_recognition/     # CNN handwriting model
│   ├── cnn_model/               # Word-level CNN classifier
│   │   ├── vocab.py             # 237-word vocabulary (names, places, relations)
│   │   ├── model.py             # 3-layer CNN architecture (~140K params)
│   │   ├── data_generator.py    # Synthetic training data with augmentation
│   │   ├── train.py             # Training pipeline with early stopping
│   │   └── inference.py         # Production inference wrapper
│   └── data/                    # Collected handwriting samples
├── Nepali_speech_to_text/       # ASR training & datasets
│   ├── src/                     # Training scripts
│   ├── notebook/                # Fine-tuning notebooks
│   └── dataset/                 # Training data & preparation scripts
└── README.md
```

## 🚀 Installation

### Prerequisites
- **Python 3.11+**
- **Git**
- CPU is sufficient — no GPU required

### Quick Start

```bash
# 1. Clone
git clone https://github.com/sandeshbhatta495/Protobytes-2.0-team-MANGALBak.git
cd Protobytes-2.0-team-MANGALBak

# 2. Create virtual environment
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start server
python main.py

# 5. Open in browser
# http://localhost:8000/app
```

> **Note:** FFmpeg is auto-configured via `imageio-ffmpeg` — no manual install needed. The Nepali Whisper model downloads automatically on first run (~1GB). Whisper tiny fallback downloads on first use (~72MB).

### Environment Variables

Create `.env.config` in the `backend/` directory (optional):

```env
HOST=0.0.0.0
PORT=8000
```

## 📡 API Endpoints

### Core Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/app` | Serve the frontend application |
| `GET` | `/health` | Health check (CNN, Whisper, ASR, Tesseract status) |
| `POST` | `/transcribe-audio` | Transcribe audio file to Nepali text |
| `POST` | `/transliterate` | Convert English text to Nepali |
| `POST` | `/correct-grammar` | Apply Nepali grammar correction |
| `POST` | `/recognize-handwriting` | Recognize handwriting (CNN + Tesseract fallback) |
| `POST` | `/generate-document` | Generate PDF from form data |
| `GET` | `/download-document/{filename}` | Download generated PDF |

### Data Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/locations` | Nepal administrative location data (cascading dropdowns) |
| `GET` | `/document-types` | List available document templates |
| `GET` | `/template/{type}` | Get form fields for a document type |
| `GET` | `/asr-status` | Check ASR model loading status |

## 💻 Usage

### 3-Step Flow
1. **Select Document** — Choose from 9 government document types
2. **Fill Form** — Use voice, handwriting, or keyboard for each field
3. **Preview & Download** — Review the generated PDF and download

### Tips for Best Results

**Voice Input:**
- Speak clearly in Nepali at normal pace
- Short phrases (3–6 seconds) give better accuracy
- 16kHz mono audio with silence trimming for efficiency

**Handwriting:**
- Write one Nepali word at a time on the canvas
- CNN returns top-k alternatives — select the correct word
- Works best with clear, large Devanagari characters

**Keyboard:**
- Type English phonetically (e.g., `namaste` → `नमस्ते`)
- Conjuncts auto-resolve (e.g., `ksha` → `क्ष`, `gya` → `ज्ञ`)

## 🔒 Security & Privacy

- No user accounts or permanent data storage
- Session-based operation — data cleared after download
- Audio files deleted immediately after transcription
- No biometric data retained
- No external API calls — all processing is local
- CORS-configured API endpoints

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow [commit message guidelines](rules%20of%20commit) (`<type>(<scope>): <description>`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

### Model Licenses
| Model | License |
|---|---|
| [Nepali ASR](https://huggingface.co/amitpant7/Nepali-Automatic-Speech-Recognition) | Apache 2.0 |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT |
| [Tesseract.js](https://github.com/naptha/tesseract.js) | Apache 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause |

## 🙏 Acknowledgments

- **amitpant7** — Fine-tuned Nepali ASR model
- **OpenAI** — Whisper speech recognition
- **HuggingFace** — Transformers library and model hosting
- **Tesseract.js** — Client-side OCR engine
- **Nepal Government** — Document format references

---

<div align="center">

**सरकारी-सारथी** — Digital Nepal Initiative 🇳🇵

Made with ❤️ for Nepal

</div>
