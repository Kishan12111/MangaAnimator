"""
Narrator Module

Text-to-speech narration generation for manga videos.
Supports Edge-TTS (Microsoft neural voices), pyttsx3, gTTS, and Coqui TTS.
"""

import asyncio
import logging
import wave
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from interfaces.base_narrator import BaseNarrator, NarrationConfig, NarrationResult

logger = logging.getLogger(__name__)


class Narrator(BaseNarrator):
    """
    Concrete implementation of TTS narration.
    
    Supports multiple TTS backends:
    - edge-tts (Microsoft neural voices, recommended)
    - pyttsx3 (offline, cross-platform)
    - gTTS (Google TTS, online)
    - Coqui TTS (local neural TTS)
    - Placeholder mode (generates silence)
    """
    
    AVAILABLE_MODELS = ['placeholder', 'elevenlabs', 'edge-tts', 'pyttsx3', 'gtts', 'coqui']
    WORDS_PER_SECOND = 2.5
    
    # High-quality Edge-TTS voices for narration 
    EDGE_TTS_VOICES = {
        'male_narrative': 'en-US-GuyNeural',
        'female_narrative': 'en-US-JennyNeural',
        'male_dramatic': 'en-US-AndrewMultilingualNeural',
        'female_expressive': 'en-US-AriaNeural',
        'male_calm': 'en-GB-RyanNeural',
        'female_warm': 'en-GB-SoniaNeural',
        'male_deep': 'en-AU-WilliamNeural',
        'female_bright': 'en-AU-NatashaNeural',
    }
    
    # ElevenLabs voice IDs for high-quality narration
    ELEVENLABS_VOICES = {
        'adam': 'pNInz6obpgDQGcFmaJgB',       # Deep, narrative male
        'daniel': 'onwK4e9ZLuTAKqWW03F9',     # Authoritative British male
        'josh': 'TxGEqnHWrfWFTfGW9XjX',       # Young, engaging male
        'charlie': 'IKne3meq5aSn9XLyUdCD',     # Casual Australian male  
        'chris': 'iP95p4xoKVk53GoZ742B',       # Casual American male
        'brian': 'nPczCjzI2devNBz1zQrb',       # Deep narrator
    }
    DEFAULT_ELEVENLABS_VOICE = 'chris'  # Great for anime recap narration
    
    def __init__(self, model_name: str = "edge-tts", voice: Optional[str] = None, api_key: Optional[str] = None):
        self._model_name = model_name
        self._engine = None
        self._model_params: Dict[str, Any] = {}
        self._default_voice = voice
        self._api_key = api_key
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the TTS model."""
        if self._model_name == "placeholder":
            logger.info("Using placeholder narrator (silence)")
            return
        
        if self._model_name == "elevenlabs":
            try:
                from elevenlabs import ElevenLabs as ElevenLabsClient
                api_key = self._api_key
                if not api_key:
                    import os
                    api_key = os.environ.get("ELEVENLABS_API_KEY")
                if not api_key:
                    raise ValueError("ElevenLabs API key not set. Add 'elevenlabs_api_key' to config.json or set ELEVENLABS_API_KEY env var.")
                self._engine = ElevenLabsClient(api_key=api_key)
                # Resolve voice name → voice_id
                if not self._default_voice:
                    self._default_voice = self.DEFAULT_ELEVENLABS_VOICE
                # If it's a preset name, resolve to ID
                if self._default_voice in self.ELEVENLABS_VOICES:
                    self._voice_id = self.ELEVENLABS_VOICES[self._default_voice]
                    self._voice_name = self._default_voice
                else:
                    # Assume it's a direct voice ID
                    self._voice_id = self._default_voice
                    self._voice_name = self._default_voice
                logger.info(f"Initialized ElevenLabs TTS with voice: {self._voice_name} ({self._voice_id})")
            except ImportError:
                logger.warning("elevenlabs not installed. Run: pip install elevenlabs. Falling back to edge-tts.")
                self._model_name = "edge-tts"
                self._initialize_model()
                return
            except Exception as e:
                logger.warning(f"ElevenLabs init failed: {e}. Falling back to edge-tts.")
                self._model_name = "edge-tts"
                self._initialize_model()
                return
        
        elif self._model_name == "edge-tts":
            try:
                import edge_tts
                # Set default voice for narration
                if not self._default_voice:
                    self._default_voice = self.EDGE_TTS_VOICES['male_narrative']
                logger.info(f"Initialized Edge-TTS with voice: {self._default_voice}")
            except ImportError:
                logger.warning("edge-tts not installed. Run: pip install edge-tts. Falling back to pyttsx3.")
                self._model_name = "pyttsx3"
                self._initialize_model()
                return
        
        elif self._model_name == "pyttsx3":
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                logger.info("Initialized pyttsx3 TTS engine")
            except Exception as e:
                logger.warning(f"Failed to initialize pyttsx3: {e}. Using placeholder.")
                self._model_name = "placeholder"
        
        elif self._model_name == "gtts":
            try:
                from gtts import gTTS
                logger.info("gTTS available")
            except ImportError:
                logger.warning("gTTS not installed. Using placeholder.")
                self._model_name = "placeholder"
        
        elif self._model_name == "coqui":
            try:
                from TTS.api import TTS
                self._engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                logger.info("Initialized Coqui TTS")
            except Exception as e:
                logger.warning(f"Failed to initialize Coqui TTS: {e}. Using placeholder.")
                self._model_name = "placeholder"
        
        else:
            logger.warning(f"Unknown model: {self._model_name}. Using placeholder.")
            self._model_name = "placeholder"
    
    def set_model(self, model_name: str, **model_params) -> None:
        """Set the TTS model."""
        self._model_name = model_name
        self._model_params = model_params
        self._initialize_model()
    
    def get_available_models(self) -> List[str]:
        """Get list of available TTS models."""
        return self.AVAILABLE_MODELS.copy()
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        if self._model_name == "edge-tts":
            return list(self.EDGE_TTS_VOICES.keys()) + list(self.EDGE_TTS_VOICES.values())
        if self._model_name == "pyttsx3" and self._engine:
            voices = self._engine.getProperty('voices')
            return [v.id for v in voices]
        return ['default']
    
    def estimate_duration(self, script: str, speed: float = 1.0) -> float:
        """Estimate narration duration for a script."""
        word_count = len(script.split())
        base_duration = word_count / self.WORDS_PER_SECOND
        return base_duration / speed
    
    def adjust_speed_for_duration(
        self,
        script: str,
        target_duration: float
    ) -> float:
        """Calculate speed needed to fit target duration."""
        estimated = self.estimate_duration(script, speed=1.0)
        
        if estimated <= target_duration:
            return 1.0
        
        # Calculate required speed (capped at 2x)
        required_speed = estimated / target_duration
        return min(required_speed, 2.0)
    
    def generate(
        self,
        script: str,
        config: Optional[NarrationConfig] = None
    ) -> NarrationResult:
        """Generate narration audio from script."""
        config = config or NarrationConfig()
        
        logger.info(f"Generating narration for {len(script.split())} words")
        
        if self._model_name == "placeholder":
            return self._generate_placeholder(script, config)
        
        if self._model_name == "elevenlabs":
            return self._generate_elevenlabs(script, config)
        
        if self._model_name == "edge-tts":
            return self._generate_edge_tts(script, config)
        
        if self._model_name == "pyttsx3":
            return self._generate_pyttsx3(script, config)
        
        if self._model_name == "gtts":
            return self._generate_gtts(script, config)
        
        if self._model_name == "coqui":
            return self._generate_coqui(script, config)
        
        return self._generate_placeholder(script, config)
    
    def generate_to_file(
        self,
        script: str,
        output_path: Path,
        config: Optional[NarrationConfig] = None
    ) -> NarrationResult:
        """Generate narration and save to file."""
        result = self.generate(script, config)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV file
        self._save_wav(result.audio_data, result.sample_rate, output_path)
        result.output_path = output_path
        
        logger.info(f"Saved narration to {output_path}")
        
        return result
    
    def _generate_placeholder(
        self, 
        script: str, 
        config: NarrationConfig
    ) -> NarrationResult:
        """Generate placeholder audio (silence with correct duration)."""
        duration = self.estimate_duration(script, config.speed)
        sample_rate = 22050
        
        # Generate silence
        samples = int(duration * sample_rate)
        audio_data = np.zeros(samples, dtype=np.float32)
        
        return NarrationResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            script=script,
            segments=self._create_segments(script, duration),
            metadata={'model': 'placeholder', 'note': 'No TTS engine available'}
        )
    
    def _generate_elevenlabs(
        self,
        script: str,
        config: NarrationConfig
    ) -> NarrationResult:
        """Generate audio using ElevenLabs (highest quality neural TTS)."""
        try:
            import tempfile
            import subprocess
            
            # Determine voice ID
            voice_id = self._voice_id
            if config.voice != 'default':
                if config.voice in self.ELEVENLABS_VOICES:
                    voice_id = self.ELEVENLABS_VOICES[config.voice]
                elif len(config.voice) > 15:  # Looks like a voice ID
                    voice_id = config.voice
            
            # ElevenLabs voice settings — balanced for adaptive tone:
            # - Moderate stability = expressive but not over-the-top
            # - High similarity = consistent voice identity
            # - Moderate style = lets the script's emotion drive delivery
            stability = 0.40       # Balanced — expressive without being manic
            similarity_boost = 0.78  # Consistent voice, slight flexibility
            style = 0.50           # Let the words carry the emotion, not the voice
            
            # Generate audio via ElevenLabs SDK
            audio_generator = self._engine.text_to_speech.convert(
                text=script,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "style": style,
                    "use_speaker_boost": True,
                },
            )
            
            # Write audio chunks to temp mp3
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_mp3 = f.name
                for chunk in audio_generator:
                    f.write(chunk)
            
            # Convert MP3 → WAV via ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name
            
            subprocess.run(
                ['ffmpeg', '-y', '-i', temp_mp3, '-ar', '24000', '-ac', '1', '-f', 'wav', temp_wav],
                capture_output=True, check=True
            )
            
            # Load WAV data
            with wave.open(temp_wav, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(n_frames)
            
            # Convert to numpy float32
            if sample_width == 2:
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
            else:
                audio_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) * 256
            
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Clean up
            Path(temp_mp3).unlink(missing_ok=True)
            Path(temp_wav).unlink(missing_ok=True)
            
            duration = len(audio_data) / sample_rate
            logger.info(f"Generated {duration:.1f}s of narration with ElevenLabs ({self._voice_name})")
            
            return NarrationResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=duration,
                script=script,
                segments=self._create_segments(script, duration),
                metadata={'model': 'elevenlabs', 'voice': self._voice_name, 'voice_id': voice_id}
            )
            
        except Exception as e:
            logger.error(f"ElevenLabs generation failed: {e}")
            logger.info("Falling back to Edge-TTS...")
            # Force a good Edge-TTS voice for the fallback (not the ElevenLabs voice name)
            fallback_config = NarrationConfig(
                voice='en-US-AndrewMultilingualNeural',
                speed=config.speed,
                pitch=config.pitch,
                volume=config.volume
            )
            result = self._generate_edge_tts(script, fallback_config)
            # Tag the result so the pipeline/frontend knows a fallback happened
            result.metadata['tts_fallback'] = True
            result.metadata['tts_fallback_reason'] = str(e)
            result.metadata['original_engine'] = 'elevenlabs'
            return result
    
    def _generate_edge_tts(
        self, 
        script: str, 
        config: NarrationConfig
    ) -> NarrationResult:
        """Generate audio using Edge-TTS (Microsoft neural voices)."""
        try:
            import edge_tts
            import tempfile
            import io
            
            # Determine voice — must be a valid Edge-TTS voice name
            # (full name like 'en-US-AndrewMultilingualNeural')
            voice = None

            # 1. Check config voice
            if config.voice and config.voice != 'default':
                if config.voice in self.EDGE_TTS_VOICES:
                    voice = self.EDGE_TTS_VOICES[config.voice]
                elif config.voice.startswith('en-'):
                    # Already a valid Edge-TTS voice identifier
                    voice = config.voice

            # 2. Check self._default_voice (only if it looks like an Edge-TTS voice)
            if not voice and self._default_voice and self._default_voice.startswith('en-'):
                voice = self._default_voice

            # 3. Fallback to a known good dramatic voice
            if not voice:
                voice = 'en-US-AndrewMultilingualNeural'
                logger.info(f"Using default Edge-TTS voice: {voice}")
            
            # Build SSML rate string from speed
            # Slightly slower than default for natural dramatic pacing
            base_rate = -5  # percent slower for natural feel
            if config.speed != 1.0:
                rate_pct = int((config.speed - 1.0) * 100) + base_rate
            else:
                rate_pct = base_rate
            rate_str = f"{rate_pct:+d}%"
            
            # Build pitch string
            pitch_str = "+0Hz"
            if config.pitch != 1.0:
                pitch_hz = int((config.pitch - 1.0) * 50)
                pitch_str = f"{pitch_hz:+d}Hz"
            
            # Generate audio using edge-tts async API
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_mp3 = f.name
            
            # Run async edge-tts in sync context
            async def _generate():
                communicate = edge_tts.Communicate(
                    text=script,
                    voice=voice,
                    rate=rate_str,
                    pitch=pitch_str,
                    volume=f"{int(config.volume * 100 - 100):+d}%"
                )
                
                # Collect word-level timing data AND audio bytes in single stream pass
                # (stream() can only be called once per Communicate instance)
                subtitles = []
                audio_chunks = []
                
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_chunks.append(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        subtitles.append({
                            'word': chunk["text"],
                            'start': chunk["offset"] / 10_000_000,  # Convert 100-ns units to seconds
                            'end': (chunk["offset"] + chunk["duration"]) / 10_000_000
                        })
                
                # Write collected audio to file
                with open(temp_mp3, 'wb') as mp3_file:
                    for audio_data in audio_chunks:
                        mp3_file.write(audio_data)
                
                return subtitles
            
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context already, create new loop
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _generate())
                        segments = future.result()
                else:
                    segments = loop.run_until_complete(_generate())
            except RuntimeError:
                segments = asyncio.run(_generate())
            
            # Convert MP3 to WAV using ffmpeg for reliable loading
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name
            
            import subprocess
            subprocess.run(
                ['ffmpeg', '-y', '-i', temp_mp3, '-ar', '24000', '-ac', '1', '-f', 'wav', temp_wav],
                capture_output=True, check=True
            )
            
            # Load WAV data
            with wave.open(temp_wav, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(n_frames)
            
            # Convert to numpy
            if sample_width == 2:
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
            else:
                audio_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) * 256
            
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Clean up
            Path(temp_mp3).unlink(missing_ok=True)
            Path(temp_wav).unlink(missing_ok=True)
            
            duration = len(audio_data) / sample_rate
            logger.info(f"Generated {duration:.1f}s of narration with Edge-TTS ({voice})")
            
            # Use word-level segments from Edge-TTS if available, else generate
            if not segments:
                segments = self._create_segments(script, duration)
            
            return NarrationResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=duration,
                script=script,
                segments=segments,
                metadata={'model': 'edge-tts', 'voice': voice}
            )
            
        except Exception as e:
            logger.error(f"Edge-TTS generation failed: {e}")
            logger.info("Falling back to pyttsx3...")
            result = self._generate_pyttsx3(script, config)
            result.metadata['tts_fallback'] = True
            result.metadata['tts_fallback_reason'] = str(e)
            result.metadata['original_engine'] = result.metadata.get('original_engine', 'edge-tts')
            return result
    
    def _generate_pyttsx3(
        self, 
        script: str, 
        config: NarrationConfig
    ) -> NarrationResult:
        """Generate audio using pyttsx3."""
        try:
            import pyttsx3
            import tempfile
            import wave
            
            # Reinitialize engine if needed (pyttsx3 can be finicky)
            if self._engine is None:
                self._engine = pyttsx3.init()
            
            # Configure engine
            if config.voice != 'default':
                try:
                    self._engine.setProperty('voice', config.voice)
                except Exception:
                    pass
            
            rate = self._engine.getProperty('rate')
            self._engine.setProperty('rate', int(rate * config.speed))
            self._engine.setProperty('volume', config.volume)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            self._engine.save_to_file(script, temp_path)
            self._engine.runAndWait()
            
            # Load the audio using wave module
            with wave.open(temp_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                raw_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
            else:  # 8-bit
                audio_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) * 256
            
            # Convert stereo to mono if needed
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # Convert to float
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            duration = len(audio_data) / sample_rate
            logger.info(f"Generated {duration:.1f}s of narration with pyttsx3")
            
            return NarrationResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=duration,
                script=script,
                segments=self._create_segments(script, duration),
                metadata={'model': 'pyttsx3'}
            )
            
        except Exception as e:
            logger.error(f"pyttsx3 generation failed: {e}")
            return self._generate_placeholder(script, config)
    
    def _generate_gtts(
        self, 
        script: str, 
        config: NarrationConfig
    ) -> NarrationResult:
        """Generate audio using gTTS."""
        try:
            from gtts import gTTS
            import tempfile
            from pydub import AudioSegment
            
            # Generate audio
            tts = gTTS(text=script, lang=config.language, slow=config.speed < 0.8)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name
            
            tts.save(temp_path)
            
            # Load and convert
            audio = AudioSegment.from_mp3(temp_path)
            
            # Adjust speed if needed
            if config.speed != 1.0:
                audio = audio.speedup(playback_speed=config.speed)
            
            # Get as numpy array
            sample_rate = audio.frame_rate
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / 32768.0  # Normalize
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            duration = len(audio_data) / sample_rate
            
            return NarrationResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=duration,
                script=script,
                segments=self._create_segments(script, duration),
                metadata={'model': 'gtts'}
            )
            
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return self._generate_placeholder(script, config)
    
    def _generate_coqui(
        self, 
        script: str, 
        config: NarrationConfig
    ) -> NarrationResult:
        """Generate audio using Coqui TTS."""
        try:
            import tempfile
            import scipy.io.wavfile as wav
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            self._engine.tts_to_file(text=script, file_path=temp_path)
            
            # Load the audio
            sample_rate, audio_data = wav.read(temp_path)
            
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            Path(temp_path).unlink(missing_ok=True)
            
            duration = len(audio_data) / sample_rate
            
            return NarrationResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=duration,
                script=script,
                segments=self._create_segments(script, duration),
                metadata={'model': 'coqui'}
            )
            
        except Exception as e:
            logger.error(f"Coqui TTS generation failed: {e}")
            return self._generate_placeholder(script, config)
    
    def _create_segments(self, script: str, duration: float) -> List[Dict[str, any]]:
        """Create approximate word timing segments."""
        words = script.split()
        if not words:
            return []
        
        time_per_word = duration / len(words)
        segments = []
        current_time = 0.0
        
        for word in words:
            segments.append({
                'word': word,
                'start': current_time,
                'end': current_time + time_per_word
            })
            current_time += time_per_word
        
        return segments
    
    def _save_wav(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        output_path: Path
    ) -> None:
        """Save audio data as WAV file."""
        # Convert to 16-bit PCM
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
