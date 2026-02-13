"""
Nepali ASR Module - Integrating the fine-tuned Nepali Whisper model
Optimized to use HuggingFace pipeline chunking for long audio.
"""
import sys
import os
import unicodedata
import re
import shutil

# Configure FFmpeg path BEFORE importing transformers
# This is critical because transformers checks for ffmpeg on import
def setup_ffmpeg():
    """Ensure FFmpeg is available in PATH for audio processing"""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        
        # The imageio_ffmpeg binary has a non-standard name, create a copy named 'ffmpeg.exe'
        ffmpeg_standard = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_standard):
            shutil.copy2(ffmpeg_exe, ffmpeg_standard)
            print(f"[FFmpeg] Created standard binary: {ffmpeg_standard}")
        
        # Prepend to PATH so it's found first
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
        
        # Also set FFMPEG_BINARY for libraries that check this
        os.environ["FFMPEG_BINARY"] = ffmpeg_standard
        
        print(f"[FFmpeg] Configured: {ffmpeg_standard}")
        return True
    except ImportError:
        print("[FFmpeg] imageio_ffmpeg not installed, trying system ffmpeg")
        return False
    except Exception as e:
        print(f"[FFmpeg] Setup error: {e}")
        return False

# Run FFmpeg setup immediately
setup_ffmpeg()

import textwrap
import numpy as np
from transformers import pipeline
import torch
import logging
import warnings
import soundfile as sf
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the Nepali speech-to-text source to path (for utilities if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Nepali_speech_to_text', 'src'))

# Try to import mp3_to_wav from utils, or define fallback
try:
    from utils import mp3_to_wav
except ImportError:
    # Fallback conversion function using pydub
    def mp3_to_wav(input_file, output_file):
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(input_file)
            audio.export(output_file, format="wav")
            print(f"Converted {input_file} to {output_file}")
            return True
        except Exception as e:
            print(f"Error converting mp3 to wav: {e}")
            return False

class NepaliASR:
    """Nepali Automatic Speech Recognition using fine-tuned Whisper model"""
    
    def __init__(self, model_name='amitpant7/Nepali-Automatic-Speech-Recognition'):
        """
        Initialize the Nepali ASR model
        
        Args:
            model_name: HuggingFace model identifier for Nepali ASR
        """
        self.model_name = model_name
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
    
    def _normalize_nepali_text(self, text):
        """
        Post-process transcription: Unicode normalize, add punctuation, clean up.
        """
        if not text:
            return ""
        
        # 1. Unicode NFC normalization
        text = unicodedata.normalize('NFC', text.strip())
        
        # 2. Basic cleanup - remove repeated words (common ASR artifact)
        words = text.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word != prev_word:
                cleaned_words.append(word)
            prev_word = word
        text = ' '.join(cleaned_words)
        
        # 3. Punctuation restoration (rule-based)
        # Add danda (।) at the end of sentences if needed
        # Look for natural sentence endings
        sentence_endings = r'(छ|छु|छन्|छौं|हो|थियो|भयो|गर्छ|गर्छु|गर्नुहोस्|हुन्छ|पर्छ|गर्दछु|भएको|गरेको)(\s|$)'
        
        # Add danda after sentence-ending verbs if not already there
        text = re.sub(sentence_endings, r'\1।\2', text)
        
        # 4. Clean up multiple dandas
        text = re.sub(r'।{2,}', '।', text)
        
        # 5. Ensure space after danda (except at end)
        text = re.sub(r'।([^\s।])', r'। \1', text)
        
        # 6. Remove trailing space before final punctuation
        text = re.sub(r'\s+([।?!])$', r'\1', text)
        
        return text.strip()
        
    def load_model(self):
        """Load the Nepali ASR model"""
        try:
            self.logger.info(f"Loading Nepali ASR model: {self.model_name}")
            # Initialize pipeline with chunking enabled for long audio files
            self.pipe = pipeline(
                "automatic-speech-recognition", 
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                chunk_length_s=30,  # Process in 30s chunks
            )
            self.logger.info("Nepali ASR model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load Nepali ASR model: {str(e)}")
            return False
    
    def transcribe_audio_file(self, audio_path):
        """
        Transcribe audio file using the Nepali model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text in Nepali
        """
        if not self.pipe:
            if not self.load_model():
                raise Exception("Failed to load ASR model")
        
        temp_wav = None
        try:
            # Convert to WAV if necessary
            if audio_path.lower().endswith('.mp3'):
                temp_wav = audio_path.replace('.mp3', '_temp.wav')
                try:
                    mp3_to_wav(audio_path, temp_wav)
                    audio_path = temp_wav
                except Exception as e:
                    self.logger.warning(f"MP3 to WAV conversion failed: {e}")
                    # If conversion fails, try passing original file to pipeline (it might handle it)
            
            # Use pipeline directly with chunking enabled
            # return_timestamps=False returns just text
            self.logger.info(f"Transcribing: {audio_path}")
            result = self.pipe(audio_path, return_timestamps=False)
            
            # Result is usually {'text': '...'} or list of chunks
            if isinstance(result, dict):
                transcription = result.get('text', '')
            elif isinstance(result, list):
                transcription = " ".join([chunk.get('text', '') for chunk in result])
            else:
                transcription = str(result)
            
            # Apply post-processing: Unicode normalization and punctuation
            transcription = self._normalize_nepali_text(transcription)
            
            return transcription.strip()
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise Exception(f"Audio transcription failed: {str(e)}")
        finally:
            # Clean up temporary file if created
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
    
    def transcribe_audio_data(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data from memory
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            Transcribed text in Nepali
        """
        if not self.pipe:
            if not self.load_model():
                raise Exception("Failed to load ASR model")
        
        temp_path = None
        try:
            # Create a temporary file for the audio data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                temp_path = tmp_file.name
            
            # Transcribe the temporary file
            transcription = self.transcribe_audio_file(temp_path)
            
            return transcription
            
        except Exception as e:
            self.logger.error(f"Audio data transcription failed: {str(e)}")
            raise Exception(f"Audio data transcription failed: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.pipe is not None,
            "language": "Nepali",
            "description": "Fine-tuned Whisper model for Nepali speech recognition"
        }

# Global instance for reuse
nepali_asr_instance = None

def get_nepali_asr():
    """Get or create the global Nepali ASR instance"""
    global nepali_asr_instance
    if nepali_asr_instance is None:
        nepali_asr_instance = NepaliASR()
    return nepali_asr_instance

# CLI usage for testing
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python nepali_asr.py audio_file.mp3')
        sys.exit(0)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    asr = NepaliASR()
    try:
        transcription = asr.transcribe_audio_file(audio_path)
        formatted_text = textwrap.fill(transcription, width=50)
        print("Transcription:")
        print(formatted_text)
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)
