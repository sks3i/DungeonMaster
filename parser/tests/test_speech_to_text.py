import io
import time
import unittest

import jax.numpy as jnp
import soundfile as sf
from gtts import gTTS

from parser.speech_to_text import SpeechToText


class TestSpeechToText(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.stt = SpeechToText()
        self.test_text = "Hello world, this is a test."

    def _create_test_audio(self, text):
        """Helper method to create synthetic audio from text.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            jnp.ndarray: Audio data as JAX numpy array
        """
        # Create synthetic audio using Google Text-to-Speech
        tts = gTTS(text)
        
        # Save audio to BytesIO buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Read audio data from buffer and convert to JAX array
        audio_data, _ = sf.read(audio_buffer)
        return jnp.array(audio_data)

    def test_speech_to_text_conversion(self):
        """Test basic speech-to-text conversion functionality."""

        # Time the speech-to-text conversion
        start_time = time.time()
        audio = self._create_test_audio(self.test_text)
        result = self.stt(audio)
        end_time = time.time()
        
        # Verify result and print timing
        self.assertEqual(result.strip(), self.test_text.strip())
        print(f"\nSpeech-to-text conversion took {end_time - start_time:.2f} seconds")
