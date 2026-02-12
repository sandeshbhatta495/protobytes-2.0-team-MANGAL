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

@app.post("/transcribe-audio")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using Nepali ASR model (with Whisper fallback)
    again falls back to gemini if both of them fail
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
async def transliterate_text(request: TransliterationRequest):
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

