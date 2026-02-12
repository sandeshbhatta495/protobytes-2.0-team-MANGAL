# Sarkari-Sarathi — AI Digital Scribe for Local Government (Nepal)

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)
![Whisper](https://img.shields.io/badge/Whisper-Nepali%20Fine--tuned-orange.svg)

**एक AI-संचालित डिजिटल स्क्राइब जसले नेपाली नागरिकहरूलाई सरकारी कागजातहरू सजिलै उत्पन्न गर्न मद्दत गर्दछ।**

*An AI-powered digital scribe helping Nepali citizens easily generate government documents.*

[Features](#-features) • [Demo](#-quick-start) • [Installation](#-installation) • [API](#-api-endpoints) • [Contributing](#-contributing)

</div>

---

## Overview

**Sarkari-Sarathi** is a comprehensive document generation system designed for Nepal's local government services. It combines:

- **Nepali Speech Recognition** — Fine-tuned Whisper model for accurate Nepali ASR
- **Handwriting Recognition** — Free-form writing input with AI-powered text extraction
- **AI Document Generation** — RAG-based template filling using Google Gemini
- **Print-Ready PDF Output** — Official government format documents

## 🎯 Features

### Multi-Modal Input Support
| Input Method | Description | Technology |
|--------------|-------------|------------|
| 🎤 **Voice Typing** | Speak in Nepali, get text | Fine-tuned Whisper ASR |
| ✍️ **Free Handwriting** | Draw/write on canvas | Gemini Vision AI |
| ⌨️ **Text Input** | Direct keyboard entry | Standard forms |

### Supported Government Documents

#### Civil Registration
- जन्म दर्ता (Birth Registration)
- मृत्यु दर्ता (Death Registration)
- विवाह दर्ता (Marriage Registration)
- सम्बन्धविच्छेद (Divorce Registration)

#### Recommendation Letters
- बसाइसराई प्रमाणपत्र (Migration Certificate)
- बसोबास प्रमाणपत्र (Residence Certificate)

#### Infrastructure & Utilities
- विद्युत जडान (Electricity Connection)
- खानेपानी जडान (Water Connection)
- बाटो पहुँच (Road Access)

### Key Highlights
- ✅ **No Login Required** — Stateless, session-based operation
- ✅ **Elder-Friendly UI** — Simple, guided step-by-step interface
- ✅ **Bilingual Support** — Nepali and English with transliteration
- ✅ **32% WER Accuracy** — Fine-tuned Nepali Whisper model
- ✅ **Offline ASR Fallback** — Works even without internet for speech recognition

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | REST API framework |
| **Python 3.8+** | Core runtime |
| **OpenAI Whisper** | Speech recognition (fallback) |
| **Fine-tuned Nepali Whisper** | Primary ASR model |
| **Google Gemini 2.0 Flash** | AI document generation & handwriting |
| **ReportLab** | PDF generation |
| **PyDub / FFmpeg** | Audio processing |
| **Transformers (HuggingFace)** | Model inference |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5 / CSS3** | Structure & styling |
| **Tailwind CSS** | Responsive design |
| **JavaScript (Vanilla)** | Interactive features |
| **Canvas API** | Handwriting input |
| **MediaRecorder API** | Voice recording |

### AI Models Used
| Model | Purpose | Source |
|-------|---------|--------|
| `amitpant7/Nepali-Automatic-Speech-Recognition` | Primary Nepali ASR | HuggingFace |
| `openai/whisper-small` | Fallback ASR | HuggingFace |
| `gemini-2.0-flash` | Document generation & OCR | Google AI |
| `Sakonii/distilbert-base-nepali` | Nepali NLP | HuggingFace |

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│     Frontend        │    │      Backend        │    │    AI Services      │
│    (HTML/JS)        │◄──►│     (FastAPI)       │◄──►│                     │
│                     │    │                     │    │  • Nepali Whisper   │
│  • Voice Recording  │    │  • Audio Processing │    │  • Gemini 2.0       │
│  • Handwriting      │    │  • RAG Templates    │    │  • Vision AI        │
│  • Form UI          │    │  • PDF Generation   │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 📁 Project Structure

```
Sarkari-Sarathi/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── nepali_asr.py            # Custom Nepali ASR module
│   ├── requirements.txt         # Python dependencies
│   ├── templates/               # Document templates (JSON)
│   │   ├── birth_registration.json
│   │   ├── death_registration.json
│   │   ├── marriage_registration.json
│   │   ├── divorce_registration.json
│   │   ├── migration_certificate.json
│   │   ├── residence_certificate.json
│   │   ├── electricity_connection.json
│   │   ├── water_connection.json
│   │   └── road_access.json
│   ├── generated_documents/     # Output PDFs
│   └── static/                  # Static assets & fonts
├── frontend/
│   ├── index.html               # Main application UI
│   └── script.js                # Frontend logic
├── Nepali_speech_to_text/       # ASR training & inference
│   ├── src/
│   │   ├── train.py             # Model training
│   │   ├── inference.py         # Inference utilities
│   │   └── utils.py             # Helper functions
│   ├── notebook/                # Jupyter notebooks
│   └── dataset/                 # Training data
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- FFmpeg (auto-installed via imageio-ffmpeg)
- CUDA-compatible GPU (recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sarkari-sarathi.git
   cd sarkari-sarathi
   ```

2. **Setup backend**
   ```bash
   cd backend
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env.config
   # Edit .env.config and add your GEMINI_API_KEY
   ```

4. **Start the server**
   ```bash
   python main.py
   ```

5. **Open the application**
   ```
   http://localhost:8000/app
   ```

### Environment Variables

Create `.env.config` in the backend directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
HOST=0.0.0.0
PORT=8000
MAX_FILE_SIZE=10485760
OUTPUT_DIR=generated_documents
LOG_LEVEL=INFO
```

## 📡 API Endpoints

### Document Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/document-types` | List available document types |
| `GET` | `/template/{type}` | Get template for document type |
| `POST` | `/generate-document` | Generate PDF document |
| `GET` | `/download-document/{filename}` | Download generated PDF |

### Audio Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcribe-audio` | Transcribe voice to Nepali text |
| `GET` | `/asr-status` | Check ASR model status |

### Text & Image Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transliterate` | Convert English to Nepali |
| `POST` | `/recognize-handwriting` | Extract text from handwriting |

### Locations
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/locations` | Get Nepal location data |

## 💻 Usage Examples

### Voice Transcription (Python)
```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe-audio",
        files={"file": f}
    )
    print(response.json()["text"])
```

### Generate Document
```python
import requests

data = {
    "document_type": "birth_registration",
    "user_data": {
        "child_name": "राम बहादुर",
        "birth_date": "२०८०-०१-१५",
        "father_name": "हरि बहादुर"
    }
}

response = requests.post(
    "http://localhost:8000/generate-document",
    json=data
)
```

## 🔊 Speech Recognition Pipeline

```
Audio Input → WebM to WAV Conversion → Nepali ASR Model → Text Output
                                              ↓
                              (Fallback) Generic Whisper
                                              ↓
                              (Fallback) Gemini Audio API
```

### Audio Preprocessing

For best results with manual audio files:
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## 🎨 UI/UX Features

- **Step-by-step guided flow** — Easy navigation for all users
- **Voice recording with visual feedback** — Real-time recording indicator
- **Free handwriting canvas** — Draw Nepali characters naturally
- **Real-time transliteration** — Type in English, get Nepali
- **Document preview** — Review before generating
- **Print-ready PDF output** — A4 format with proper letterhead
- **Service rating system** — Feedback collection

## 🔒 Security & Privacy

- No permanent data storage
- Session-based operation only
- Automatic file cleanup after download
- No biometric data processing
- Secure API endpoints with CORS

## 🧑‍💻 Development

### Adding New Document Templates

1. Create JSON in `backend/templates/`:
```json
{
  "name": "Document Name",
  "name_ne": "कागजात नाम",
  "category": "civil_registration",
  "required_fields": ["field1", "field2"],
  "optional_fields": ["field3"],
  "content_template": "Template with {placeholders}",
  "instructions": ["Step 1", "Step 2"]
}
```

2. Restart the server to load new template

### Running Tests
```bash
cd backend
python -m pytest tests/
```

## 🐳 Deployment

### Docker
```bash
docker build -t sarkari-sarathi .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key sarkari-sarathi
```

### Production Checklist
- [ ] Use HTTPS with SSL
- [ ] Enable rate limiting
- [ ] Set up monitoring/logging
- [ ] Configure proper CORS origins
- [ ] Use production WSGI server (gunicorn)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Model Licenses
- [Whisper](https://huggingface.co/openai/whisper-small) — MIT License
- [DistilBERT Nepali](https://huggingface.co/Sakonii/distilbert-base-nepali) — Apache 2.0
- [Nepali ASR](https://huggingface.co/amitpant7/Nepali-Automatic-Speech-Recognition) — Apache 2.0

## 🙏 Acknowledgments

- **OpenAI** — Whisper speech recognition model
- **Google** — Gemini AI for document generation
- **HuggingFace** — Transformers library and model hosting
- **amitpant7** — Fine-tuned Nepali ASR model
- **Sakonii** — DistilBERT Nepali model
- **Nepal Government** — Document format references

---

<div align="center">

**सरकारी-सारथी** — Digital Nepal Initiative 🇳🇵

Made with ❤️ for Nepal

</div>
