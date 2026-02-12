import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
# Load environment variables from .env.config file (since .env is used as venv dir)
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
config_path = os.path.join(BASE_DIR, ".env.config")
load_dotenv(dotenv_path=config_path)

if not os.getenv("GEMINI_API_KEY"):
    import logging as _log
    _log.warning("GEMINI_API_KEY not found in .env.config, trying default load_dotenv")
    # Fallback to default check (might load from system env)
    load_dotenv()
import tempfile
from typing import Optional, Dict, Any
import aiofiles
import whisper
import torch
from datetime import datetime
import google.generativeai as genai
from pydantic import BaseModel
import asyncio
import textwrap
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sarkari-Sarathi API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
whisper_model = None
templates = {}
gemini_model = None
nepali_asr = None  # Our custom Nepali ASR instance

class DocumentRequest(BaseModel):
    document_type: str
    user_data: Dict[str, Any]
    language: str = "ne"

class TransliterationRequest(BaseModel):
    text: str
    from_lang: str = "en"
    to_lang: str = "ne"

async def initialize_models():
    global whisper_model, gemini_model, nepali_asr
    
    # Load templates first so they are always available even if models fail
    await load_templates()
    logger.info(f"Loaded {len(templates)} document templates")
    
    # Load Whisper model (as fallback)
    try:
        whisper_model = whisper.load_model("small")
        logger.info("Generic Whisper model loaded as fallback")
    except Exception as e:
        logger.warning(f"Failed to load generic Whisper model: {e}")
    
    # Initialize our custom Nepali ASR
    try:
        nepali_asr = get_nepali_asr()
        if nepali_asr.load_model():
            logger.info("Nepali ASR model loaded successfully")
        else:
            logger.warning("Failed to load Nepali ASR model, will use fallback")
    except Exception as e:
        logger.warning(f"Failed to initialize Nepali ASR: {e}")
    
    # Configure Gemini (you'll need to set GEMINI_API_KEY in environment)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Gemini model configured")
    else:
        logger.warning("GEMINI_API_KEY not found in environment")

async def load_templates():
    global templates
    template_dir = os.path.join(BASE_DIR, "templates")
    if not os.path.exists(template_dir):
        logger.warning(f"Template directory not found: {template_dir}")
        return
    for filename in os.listdir(template_dir):
        if filename.endswith('.json'):
            try:
                path = os.path.join(template_dir, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    templates[filename[:-5]] = json.load(f)
                logger.info(f"Loaded template: {filename}")
            except Exception as e:
                logger.error(f"Failed to load template {filename}: {e}")

@app.on_event("startup")
async def startup_event():
    await initialize_models()

@app.get("/")
async def root():
    return {"message": "Sarkari-Sarathi API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint with model status"""
    return {
        "status": "ok",
        "models": {
            "nepali_asr": {
                "loaded": nepali_asr is not None and nepali_asr.pipe is not None,
                "model_name": nepali_asr.model_name if nepali_asr else None
            },
            "whisper": {
                "loaded": whisper_model is not None
            },
            "gemini": {
                "loaded": gemini_model is not None,
                "api_key_set": bool(os.getenv("GEMINI_API_KEY"))
            }
        },
        "templates_loaded": len(templates)
    }


