from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
import uvicorn
import os
from dotenv import load_dotenv
import json
import secrets
from collections import defaultdict
import time

# Configure FFmpeg path from imageio_ffmpeg before importing audio libraries
try:
    import imageio_ffmpeg
    import shutil
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    # Create a copy named ffmpeg.exe if it doesn't exist (libraries look for 'ffmpeg')
    ffmpeg_standard = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_standard) and os.path.exists(ffmpeg_exe):
        shutil.copy2(ffmpeg_exe, ffmpeg_standard)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    # Also set FFMPEG_BINARY for pydub
    os.environ["FFMPEG_BINARY"] = ffmpeg_standard if os.path.exists(ffmpeg_standard) else ffmpeg_exe
except ImportError:
    pass  # FFmpeg should be in system PATH

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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
import textwrap
import logging

# Import our custom Nepali ASR module
from nepali_asr import get_nepali_asr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sarkari-Sarathi API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# Security Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5500,http://localhost:3000").split(",")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Global rate limiter
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW) -> bool:
        now = time.time()
        window_start = now - window
        
        # Clean old requests
        self.requests[client_ip] = [ts for ts in self.requests[client_ip] if ts > window_start]
        
        # Check limit
        if len(self.requests[client_ip]) >= max_requests:
            return False
        
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter()

# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com fonts.googleapis.com; font-src 'self' fonts.gstatic.com"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Serve entire frontend folder at /app so index.html and script.js load reliably
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"Frontend mounted at /app from {FRONTEND_DIR}")

# API Key Authentication Dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if API_SECRET_KEY is configured"""
    if API_SECRET_KEY and API_SECRET_KEY.strip():
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required. Include X-API-Key header."
            )
        if not secrets.compare_digest(x_api_key, API_SECRET_KEY):
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
    return True

# Rate Limiting Dependency
async def check_rate_limit_dependency(request: Request):
    """Check global rate limit for the request"""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )
    return True

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

class GrammarCorrectionRequest(BaseModel):
    text: str
    context: Optional[str] = None

class HandwritingRequest(BaseModel):
    image_data: str  # Base64 encoded image

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

@app.post("/transcribe-audio")
async def transcribe_audio(
    audio: UploadFile = File(...),
    _rate_limit: bool = Depends(check_rate_limit_dependency),
    _api_key: bool = Depends(verify_api_key)
):
    """
    Transcribe audio using Nepali ASR model (with Whisper fallback)
    """
    import subprocess
    
    # Determine suffix from content type
    content_type = audio.content_type or ''
    suffix = ".wav"
    if 'webm' in content_type:
        suffix = ".webm"
    elif 'ogg' in content_type:
        suffix = ".ogg"
    elif 'mp4' in content_type or 'mpeg' in content_type:
        suffix = ".mp4"

    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    wav_path = tmp_file_path
    
    try:
        # Convert to WAV 16kHz mono if not already WAV (browser typically sends webm)
        if suffix != ".wav":
            wav_path = tmp_file_path.replace(suffix, "_converted.wav")
            try:
                # Get ffmpeg path
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError:
                    ffmpeg_exe = "ffmpeg"
                
                result = subprocess.run(
                    [ffmpeg_exe, '-y', '-i', tmp_file_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav_path],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    logger.error(f"FFmpeg conversion error: {result.stderr}")
                    wav_path = tmp_file_path  # Try with original file
                else:
                    logger.info(f"Converted {suffix} to WAV successfully")
            except Exception as conv_err:
                logger.warning(f"Audio conversion failed: {conv_err}, trying with original file")
                wav_path = tmp_file_path
        
        transcription = None
        language_detected = "ne"
        model_used = None  # Track which model actually produced the transcription
        
        # Try Nepali ASR first
        if nepali_asr and nepali_asr.pipe is not None:
            try:
                logger.info("Using Nepali ASR model for transcription")
                transcription = nepali_asr.transcribe_audio_file(wav_path)
                model_used = "nepali_asr"
                logger.info(f"Nepali ASR transcription successful: {transcription[:50]}...")
            except Exception as e:
                logger.warning(f"Nepali ASR failed: {e}")
        
        # Fallback to generic Whisper if Nepali ASR fails
        if not transcription and whisper_model:
            try:
                logger.info("Falling back to generic Whisper model")
                result = whisper_model.transcribe(wav_path)
                transcription = result["text"]
                language_detected = result.get("language", "ne")
                model_used = "whisper_fallback"
                logger.info(f"Whisper transcription successful: {transcription[:50]}...")
            except Exception as e:
                logger.error(f"Whisper fallback also failed: {e}")
        
        # Fallback to Gemini if both ASR models fail
        if not transcription and gemini_model:
            try:
                logger.info("Falling back to Gemini for audio transcription")
                # Read the wav file and send to Gemini
                audio_file_path = wav_path if os.path.exists(wav_path) else tmp_file_path
                
                # Gemini needs audio as a file upload or inline data
                # Use the google.generativeai File API for better handling
                import base64 as b64
                
                with open(audio_file_path, 'rb') as af:
                    audio_bytes = af.read()
                
                audio_b64 = b64.b64encode(audio_bytes).decode('utf-8')
                
                # Use Gemini's audio understanding with proper inline_data format
                prompt = """यो अडियो फाइलमा नेपाली भाषामा बोलिएको छ। कृपया सुन्नुहोस् र शुद्ध नेपाली युनिकोडमा लेख्नुहोस्।
केवल बोलिएको पाठ मात्र लेख्नुहोस्, अरू केही नलेख्नुहोस्।"""
                
                # Create inline data part for Gemini
                audio_part = {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_b64
                    }
                }
                
                logger.info(f"Sending {len(audio_bytes)} bytes of audio to Gemini...")
                response = await asyncio.to_thread(gemini_model.generate_content, [prompt, audio_part])
                transcription = response.text.strip() if response.text else ""
                
                if transcription:
                    model_used = "gemini_audio"
                    logger.info(f"Gemini audio transcription: {transcription[:50]}...")
                else:
                    logger.warning("Gemini returned empty transcription")
            except Exception as e:
                logger.warning(f"Gemini audio fallback also failed: {e}", exc_info=True)
        
        if not transcription:
            raise HTTPException(status_code=500, detail="सबै ट्रान्सक्रिप्शन विधिहरू असफल भए। कृपया स्पष्ट रूपमा बोल्नुहोस् र पुनः प्रयास गर्नुहोस्।")
        
        # Post-process: correct Nepali grammar
        corrected_transcription = await correct_nepali_grammar(transcription)
        
        # Clean up temporary files
        for f in [tmp_file_path, wav_path]:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass
        
        return {
            "transcription": corrected_transcription,
            "raw_transcription": transcription,
            "language": language_detected,
            "model_used": model_used,
            "grammar_corrected": corrected_transcription != transcription
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up temporary files if they exist
        for f in [tmp_file_path, wav_path]:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/asr-status")
async def get_asr_status():
    """Get the status of ASR models"""
    global nepali_asr, whisper_model
    
    status = {
        "nepali_asr": {
            "available": nepali_asr is not None and nepali_asr.pipe is not None,
            "model": nepali_asr.model_name if nepali_asr else None,
            "device": nepali_asr.device if nepali_asr else None
        },
        "whisper_fallback": {
            "available": whisper_model is not None,
            "model": "openai/whisper-small" if whisper_model else None
        }
    }
    
    return status

@app.post("/transliterate")
async def transliterate_text(
    request: TransliterationRequest,
    _rate_limit: bool = Depends(check_rate_limit_dependency)
):
    """Translate/Transliterate text using Gemini AI with offline fallback"""
    try:
        if gemini_model and request.text.strip():
            # Use Gemini for translation
            if request.from_lang == "en" and request.to_lang == "ne":
                prompt = f"""Translate the following English text to Nepali. Only return the Nepali translation, nothing else.

English: {request.text}

Nepali:"""
            elif request.from_lang == "ne" and request.to_lang == "en":
                prompt = f"""Translate the following Nepali text to English. Only return the English translation, nothing else.

Nepali: {request.text}

English:"""
            else:
                # Transliterate English to Nepali phonetically
                prompt = f"""Transliterate the following English text to Nepali script (phonetically). Only return the Nepali script, nothing else.

English: {request.text}

Nepali:"""
            
            response = gemini_model.generate_content(prompt)
            nepali_text = response.text.strip()
            logger.info(f"Translation: '{request.text}' -> '{nepali_text}'")
            return {"original_text": request.text, "transliterated_text": nepali_text}
        else:
            # No Gemini model — use offline transliteration
            result = offline_transliterate(request.text)
            return {"original_text": request.text, "transliterated_text": result, "method": "offline"}
    except Exception as e:
        logger.error(f"Translation error: {e}")
        # Fallback to offline transliteration instead of returning original text
        try:
            result = offline_transliterate(request.text)
            return {"original_text": request.text, "transliterated_text": result, "method": "offline_fallback"}
        except:
            return {"original_text": request.text, "transliterated_text": request.text}


def offline_transliterate(text: str) -> str:
    """
    Offline English-to-Nepali phonetic transliteration.
    Maps Roman/English characters to Devanagari equivalents.
    """
    # Multi-character mappings (check longer sequences first)
    multi_map = {
        'shri': 'श्री', 'shr': 'श्र', 'ksh': 'क्ष', 'tra': 'त्र', 'gya': 'ज्ञ',
        'chh': 'छ', 'thh': 'ठ', 'dhh': 'ढ', 'shh': 'ष',
        'kha': 'खा', 'gha': 'घा', 'cha': 'चा', 'chha': 'छा',
        'jha': 'झा', 'tha': 'था', 'dha': 'धा', 'pha': 'फा',
        'bha': 'भा', 'sha': 'शा',
        'kh': 'ख', 'gh': 'घ', 'ng': 'ङ',
        'ch': 'च', 'jh': 'झ', 'ny': 'ञ',
        'th': 'थ', 'dh': 'ध', 'ph': 'फ',
        'bh': 'भ', 'sh': 'श',
        'aa': 'ा', 'ee': 'ी', 'oo': 'ू', 'ai': 'ै', 'au': 'ौ',
        'ou': 'ौ', 'ei': 'ै',
    }
    
    # Single character vowel mappings (standalone & matra)
    vowel_standalone = {
        'a': 'अ', 'i': 'इ', 'u': 'उ', 'e': 'ए', 'o': 'ओ',
    }
    vowel_matra = {
        'a': '', 'i': 'ि', 'u': 'ु', 'e': 'े', 'o': 'ो',
    }
    
    # Single consonant mappings
    consonant_map = {
        'k': 'क', 'g': 'ग', 'c': 'च', 'j': 'ज', 't': 'त',
        'd': 'द', 'n': 'न', 'p': 'प', 'b': 'ब', 'm': 'म',
        'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
        's': 'स', 'h': 'ह', 'f': 'फ', 'z': 'ज़', 'x': 'क्स',
        'q': 'क',
    }
    
    # Common word translations for form fields
    word_map = {
        'name': 'नाम', 'first name': 'पहिलो नाम', 'last name': 'थर',
        'father': 'बुबा', 'mother': 'आमा', 'address': 'ठेगाना',
        'phone': 'फोन', 'email': 'इमेल', 'date': 'मिति',
        'district': 'जिल्ला', 'province': 'प्रदेश', 'ward': 'वडा',
        'municipality': 'नगरपालिका', 'village': 'गाउँ', 'city': 'शहर',
        'male': 'पुरुष', 'female': 'महिला', 'age': 'उमेर',
        'birth': 'जन्म', 'death': 'मृत्यु', 'marriage': 'विवाह',
        'husband': 'पति', 'wife': 'पत्नी', 'son': 'छोरा', 'daughter': 'छोरी',
        'yes': 'हो', 'no': 'होइन', 'number': 'नम्बर',
        'citizenship': 'नागरिकता', 'certificate': 'प्रमाणपत्र',
        'application': 'निवेदन', 'registration': 'दर्ता',
        'signature': 'हस्ताक्षर', 'photo': 'फोटो',
        'occupation': 'पेशा', 'religion': 'धर्म', 'nationality': 'राष्ट्रियता',
        'country': 'देश', 'nepal': 'नेपाल',
        'road': 'सडक', 'water': 'पानी', 'electricity': 'बिजुली',
        'school': 'विद्यालय', 'hospital': 'अस्पताल',
        'government': 'सरकार', 'office': 'कार्यालय',
    }
    
    text_lower = text.lower().strip()
    
    # Check if whole text matches a known word/phrase
    if text_lower in word_map:
        return word_map[text_lower]
    
    # Character-by-character transliteration
    result = []
    i = 0
    after_consonant = False
    
    while i < len(text):
        char = text[i].lower()
        matched = False
        
        # Skip spaces and punctuation
        if char == ' ':
            result.append(' ')
            after_consonant = False
            i += 1
            continue
        if char in '.,;:!?()[]{}"\'-/\\@#$%^&*+=<>|~`':
            result.append(char)
            after_consonant = False
            i += 1
            continue
        if char.isdigit():
            # Convert to Devanagari digits
            devanagari_digits = '०१२३४५६७८९'
            result.append(devanagari_digits[int(char)])
            after_consonant = False
            i += 1
            continue
        
        # Try multi-character matches (longest first)
        for length in range(4, 1, -1):
            substr = text[i:i+length].lower()
            if substr in multi_map:
                result.append(multi_map[substr])
                after_consonant = substr[-1] not in 'aeiou'
                i += length
                matched = True
                break
        
        if matched:
            continue
        
        # Single character
        if char in consonant_map:
            if after_consonant:
                result.append('्')  # halant to join consonants
            result.append(consonant_map[char])
            after_consonant = True
            i += 1
        elif char in vowel_standalone:
            if after_consonant:
                result.append(vowel_matra[char])
            else:
                result.append(vowel_standalone[char])
            after_consonant = False
            i += 1
        else:
            result.append(char)
            after_consonant = False
            i += 1
    
    return ''.join(result)

class HandwritingRequest(BaseModel):
    image: str  # Base64 encoded image data

class GrammarCorrectionRequest(BaseModel):
    text: str
    context: str = ""  # optional field label / context


async def correct_nepali_grammar(text: str, context: str = "") -> str:
    """Post-process Nepali text for grammar correction using Gemini."""
    if not gemini_model or not text or not text.strip():
        return text
    try:
        ctx_hint = f" (यो फिल्ड: {context})" if context else ""
        prompt = (
            "तपाईंलाई नेपाली पाठ दिइएको छ। कृपया यसलाई शुद्ध नेपाली व्याकरणमा सच्याउनुहोस्।\n"
            "- मात्रा, हलन्त, र विसर्ग ठीक गर्नुहोस्\n"
            "- शब्द क्रम र विभक्ति मिलाउनुहोस्\n"
            "- अर्थ नबिगार्नुहोस्, केवल व्याकरण सच्याउनुहोस्\n"
            "- केवल सच्याइएको नेपाली पाठ मात्र फर्काउनुहोस्, अरू केही नलेख्नुहोस्\n"
            f"{ctx_hint}\n\n"
            f"पाठ: {text}\n\n"
            "शुद्ध पाठ:"
        )
        # Run synchronous Gemini call in threadpool to avoid blocking the event loop
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        corrected = response.text.strip()
        # Sanity check: if Gemini returned something wildly different or empty, keep original
        if not corrected or len(corrected) > len(text) * 3:
            return text
        logger.info(f"Grammar correction: '{text[:40]}' -> '{corrected[:40]}'")
        return corrected
    except Exception as e:
        logger.warning(f"Grammar correction failed, returning original: {e}")
        return text


# Simple in-memory rate limiter for grammar correction endpoint
_grammar_rate_limit_store: Dict[str, list] = {}
GRAMMAR_RATE_LIMIT_MAX_REQUESTS = 10  # Max requests per window
GRAMMAR_RATE_LIMIT_WINDOW_SECONDS = 60  # Time window in seconds
MAX_GRAMMAR_TEXT_LENGTH = 5000  # Maximum text length for grammar correction


def check_rate_limit(client_ip: str) -> bool:
    """Check if client is within rate limit. Returns True if allowed, False if limited."""
    now = datetime.now().timestamp()
    window_start = now - GRAMMAR_RATE_LIMIT_WINDOW_SECONDS
    
    if client_ip not in _grammar_rate_limit_store:
        _grammar_rate_limit_store[client_ip] = []
    
    # Clean old entries
    _grammar_rate_limit_store[client_ip] = [
        ts for ts in _grammar_rate_limit_store[client_ip] if ts > window_start
    ]
    
    # Check limit
    if len(_grammar_rate_limit_store[client_ip]) >= GRAMMAR_RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Record this request
    _grammar_rate_limit_store[client_ip].append(now)
    return True


@app.post("/correct-grammar")
async def correct_grammar_endpoint(
    request: GrammarCorrectionRequest,
    req: Request,
    _rate_limit: bool = Depends(check_rate_limit_dependency),
    _api_key: bool = Depends(verify_api_key)
):
    """Correct Nepali grammar in the given text using Gemini AI"""
    # Get client IP for rate limiting
    client_ip = req.client.host if req.client else "unknown"
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Too many requests. Please wait before trying again."
        )
    
    # Validate text length
    if len(request.text) > MAX_GRAMMAR_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {MAX_GRAMMAR_TEXT_LENGTH} characters allowed."
        )
    
    corrected = await correct_nepali_grammar(request.text, request.context)
    return {"original": request.text, "corrected": corrected}


# currently gemini
# Local model to be made
# Needs lots of fixing
@app.post("/recognize-handwriting")
async def recognize_handwriting(request: HandwritingRequest):
    """Recognize handwritten text from canvas image using Gemini Vision"""
    try:
        if not gemini_model:
            raise HTTPException(status_code=503, detail="AI model not available. Please set GEMINI_API_KEY in .env.config")
        
        # Extract base64 image data (remove data:image/png;base64, prefix if present)
        image_data = request.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        import base64
        from PIL import Image
        import io
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        logger.info(f"Received image: mode={image.mode}, size={image.size}")
        
        # Verify image has actual content (not blank)
        # Check for any pixel with significant alpha (drawn content)
        has_content = False
        img_array = list(image.getdata())
        for pixel in img_array:
            if len(pixel) == 4:  # RGBA
                r, g, b, a = pixel
                # Any pixel with alpha > 10 means something was drawn
                if a > 10:
                    has_content = True
                    break
            elif len(pixel) == 3:  # RGB
                r, g, b = pixel
                # For non-RGBA, check for non-white pixels
                if r < 240 or g < 240 or b < 240:
                    has_content = True
                    break
        
        if not has_content:
            logger.warning("Canvas appears empty - no drawn content detected")
            return {"text": "", "success": False, "detail": "Canvas appears empty"}
        
        # Convert RGBA to RGB with white background for better recognition
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image using alpha channel as mask
            background.paste(image, mask=image.split()[3])
            image = background
            logger.info("Converted RGBA to RGB with white background")
        
        # Use Gemini Vision to recognize handwriting — focused Nepali prompt
        prompt = """यो छविमा हातले लेखिएको नेपाली (देवनागरी लिपि) पाठ छ।
कृपया छविमा देखिएको सबै पाठ ध्यानपूर्वक पढ्नुहोस् र लेख्नुहोस्।
नियमहरू:
1. केवल पढिएको पाठ मात्र फर्काउनुहोस्
2. यदि देवनागरी लिपि हो भने नेपाली युनिकोडमा लेख्नुहोस्  
3. यदि English अक्षर देखिन्छ भने English मा नै राख्नुहोस्
4. कुनै व्याख्या वा थप टिप्पणी नलेख्नुहोस्
5. शुद्ध नेपाली व्याकरणमा लेख्नुहोस्"""
        
        logger.info("Sending image to Gemini for recognition...")
        response = await asyncio.to_thread(gemini_model.generate_content, [prompt, image])
        recognized_text = response.text.strip()
        
        logger.info(f"Gemini raw response: {recognized_text[:200] if recognized_text else '(empty)'}")
        
        # Remove any markdown formatting Gemini might add
        recognized_text = recognized_text.strip('`').strip('*').strip()
        if recognized_text.startswith('```'):
            recognized_text = recognized_text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        
        logger.info(f"Handwriting recognition result: {recognized_text[:100] if recognized_text else '(empty)'}")
        
        return {"text": recognized_text, "success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Handwriting recognition error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@app.post("/generate-document")
async def generate_document(
    request: DocumentRequest,
    _rate_limit: bool = Depends(check_rate_limit_dependency),
    _api_key: bool = Depends(verify_api_key)
):
    if request.document_type not in templates:
        raise HTTPException(status_code=400, detail="Document template not found")
    
    template = templates[request.document_type]
    
    # Don't block on missing optional fields - just warn
    missing_fields = []
    for field in template.get("required_fields", []):
        if field not in request.user_data or not str(request.user_data.get(field, '')).strip():
            missing_fields.append(field)
    
    if missing_fields:
        logger.warning(f"Missing fields for {request.document_type}: {missing_fields}")
        # Only block if more than half are missing
        if len(missing_fields) > len(template.get('required_fields', [])) / 2:
            raise HTTPException(
                status_code=400, 
                detail=f"धेरै आवश्यक फिल्डहरू छुटेकाछन्: {', '.join(missing_fields)}"
            )
    
    try:
        # Generate document content using template
        document_content = fill_template(template, request.user_data)
        
        # Generate PDF
        pdf_path = await generate_pdf(document_content, request.document_type, request.user_data)
        
        return {
            "document_id": f"{request.document_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "pdf_path": pdf_path,
            "content": document_content
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"दस्तावेज उत्पन्न गर्न सकेन: {str(e)}")

def fill_template(template: Dict, user_data: Dict) -> str:
    """Fill template with user data"""
    content = template.get("content", "")
    if not content:
        # Build a basic content from user data
        lines = []
        for key, value in user_data.items():
            if value:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    # Replace placeholders with actual data
    for key, value in user_data.items():
        placeholder = f"{{{key}}}"
        content = content.replace(placeholder, str(value) if value else '')
    
    # Add current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    content = content.replace("{date}", current_date)
    content = content.replace("{date_bs}", convert_to_bikram_sambat(current_date))
    
    # Remove any remaining unresolved placeholders like {something}
    import re
    content = re.sub(r'\{[a-zA-Z_]+\}', '..........', content)
    
    return content

def convert_to_bikram_sambat(date_gregorian: str) -> str:
    """Convert Gregorian date to Bikram Sambat (approximate)"""
    try:
        dt = datetime.strptime(date_gregorian, "%Y-%m-%d")
        # Approximate BS = AD + 56 years 8 months
        bs_year = dt.year + 56
        bs_month = dt.month + 8
        bs_day = dt.day + 16
        if bs_day > 30:
            bs_day -= 30
            bs_month += 1
        if bs_month > 12:
            bs_month -= 12
            bs_year += 1
        return f"{bs_year}-{bs_month:02d}-{bs_day:02d}"
    except:
        return date_gregorian

# Register Nepali font once at module level
_nepali_font_registered = False
_nepali_font_name = 'Helvetica'

def _ensure_nepali_font():
    global _nepali_font_registered, _nepali_font_name
    if _nepali_font_registered:
        return _nepali_font_name
    try:
        nepali_font_path = os.path.join(BASE_DIR, 'static', 'fonts', 'NotoSansDevanagari-Regular.ttf')
        if os.path.exists(nepali_font_path):
            pdfmetrics.registerFont(TTFont('NotoSansDevanagari', nepali_font_path))
            _nepali_font_name = 'NotoSansDevanagari'
            logger.info(f"Nepali font registered from {nepali_font_path}")
        else:
            logger.warning(f"Nepali font not found at {nepali_font_path}")
    except Exception as e:
        logger.warning(f"Could not register Nepali font: {e}")
    _nepali_font_registered = True
    return _nepali_font_name

async def generate_pdf(content: str, document_type: str, user_data: Dict) -> str:
    """Generate PDF document with proper Nepali layout"""
    output_dir = os.path.join(BASE_DIR, "generated_documents")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{document_type}_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)
    
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    font_name = _ensure_nepali_font()
    
    # Nepal Government Header
    c.setFont(font_name, 14)
    c.drawCentredString(width / 2, height - 0.7 * inch, "नेपाल सरकार")
    
    # Municipality from user data or default
    municipality = user_data.get('municipality', '')
    district = user_data.get('district', '')
    province = user_data.get('province', '')
    ward = user_data.get('ward', '')
    
    c.setFont(font_name, 16)
    header_text = municipality if municipality else "स्थानीय तह"
    c.drawCentredString(width / 2, height - 1.0 * inch, header_text)
    
    c.setFont(font_name, 11)
    if district and province:
        c.drawCentredString(width / 2, height - 1.25 * inch, f"{district}, {province}")
    
    if ward:
        c.setFont(font_name, 11)
        c.drawCentredString(width / 2, height - 1.45 * inch, f"वडा नं. {ward} को कार्यालय")
    
    # Horizontal line
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.line(0.75 * inch, height - 1.6 * inch, width - 0.75 * inch, height - 1.6 * inch)
    
    # Subject
    c.setFont(font_name, 12)
    subject = f"विषय: {get_document_subject(document_type)}"
    c.drawString(inch, height - 1.9 * inch, subject)
    
    # Date on the right TOTTTTTTTT
    date_bs = convert_to_bikram_sambat(datetime.now().strftime('%Y-%m-%d'))
    c.setFont(font_name, 10)
    c.drawRightString(width - inch, height - 1.9 * inch, f"मिति: {date_bs}")

    # Body contenttt
    c.setFont(font_name, 11)
    y_position = height - 2.3 * inch
    
    lines = content.split('\n')
    for line in lines:
        if line.strip():
            # Nepali text wrapping: use ~55 chars per line for proper fit
            wrapped_lines = textwrap.wrap(line, width=55) if len(line) > 55 else [line]
            for wrapped_line in wrapped_lines:
                if y_position < 2.5 * inch:
                    # New page
                    c.showPage()
                    c.setFont(font_name, 11)
                    y_position = height - inch
                c.drawString(inch, y_position, wrapped_line)
                y_position -= 0.22 * inch
        else:
            y_position -= 0.12 * inch
    
    # Signature at bottom
    if y_position < 3.5 * inch:
        c.showPage()
        c.setFont(font_name, 11)
        y_position = height - inch
    
    sig_y = 2.2 * inch
    c.setFont(font_name, 10)
    
    # Left applicant
    c.drawString(inch, sig_y + 0.3 * inch, "..............................")
    c.drawString(inch, sig_y, "निवेदकको हस्ताक्षर")
    
    # Right authority  
    c.drawString(width - 3 * inch, sig_y + 0.3 * inch, "..............................")
    c.drawString(width - 3 * inch, sig_y, "प्रमाणिकरण अधिकारी")
    
    # Bottom line
    c.setFont(font_name, 8)
    c.drawCentredString(width / 2, 0.5 * inch, "यो दस्तावेज सरकारी-सारथी AI Digital Scribe मार्फत उत्पन्न गरिएको हो।")
    
    c.save()
    logger.info(f"PDF generated: {filepath}")
    return filepath

def get_document_subject(document_type: str) -> str:
    """Get Nepali subject line for document type"""
    subjects = {
        "birth_registration": "जन्म दर्ताको निवेदन",
        "death_registration": "मृत्यु दर्ताको निवेदन", 
        "marriage_registration": "विवाह दर्ताको निवेदन",
        "migration_certificate": "बसाइसराई प्रमाणपत्रको निवेदन",
        "residence_certificate": "बसोबास प्रमाणपत्रको निवेदन",
        "electricity_connection": "विद्युत जडानको निवेदन",
        "water_connection": "खानेपानी जडानको निवेदन",
        "road_access": "बाटो पहुँचको निवेदन"
    }
    return subjects.get(document_type, "निवेदन")

@app.get("/download-document/{filename}")
async def download_document(filename: str):
    """Download generated document"""
    file_path = os.path.join(BASE_DIR, "generated_documents", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    return FileResponse(file_path, media_type='application/pdf', filename=filename)

@app.get("/templates")
async def list_templates():
    """List loaded template keys (for debugging)."""
    return {"loaded": list(templates.keys()), "count": len(templates)}


@app.get("/document-types")
async def get_document_types():
    """Get available document types"""
    return {
        "document_types": list(templates.keys()),
        "categories": {
            "civil_registration": ["birth_registration", "death_registration", "marriage_registration", "divorce_registration"],
            "recommendation": ["migration_certificate", "residence_certificate"],
            "infrastructure": ["electricity_connection", "water_connection", "road_access"]
        }
    }

@app.get("/locations")
async def get_locations():
    """Get Nepal administrative division data"""
    try:
        location_file = os.path.join(BASE_DIR, "locations.json")
        if os.path.exists(location_file):
            with open(location_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"error": "Locations file not found"}
    except Exception as e:
        logger.error(f"Error serving locations: {e}")
        raise HTTPException(status_code=500, detail="Error loading location data")

@app.get("/template/{document_type}")
async def get_template(document_type: str):
    """Get pattern definition for a specific document type"""
    if document_type not in templates:
        raise HTTPException(status_code=404, detail="Template not found")
    return templates[document_type]

if __name__ == "__main__":
    import socket
    import sys
    
    # Find available port starting from 8000
    def find_available_port(start_port=8000):
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    s.close()
                    return port
            except OSError:
                continue
        return None
    
    available_port = find_available_port()
    if available_port:
        print(f"Starting server on port {available_port}")
        print(f"API docs at: http://localhost:{available_port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=available_port)
    else:
        print("❌ No available ports found in range 8000-8010")
        print("Please close some applications and try again")
        sys.exit(1)
