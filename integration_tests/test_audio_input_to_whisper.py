import unittest
import time

from audio_io.audio import AudioRecorder
from parser.speech_to_text import SpeechToText


class TestAudioInputToWhisper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recorder = AudioRecorder()
        self.stt = SpeechToText()

    def test_live_speech_to_text(self):
        """Test recording and transcription pipeline.
        
        This is an interactive test that requires user input:
        1. Hold spacebar and speak
        2. Release spacebar to process
        """
        print("\nStarting live speech test...")
        print("Hold spacebar and speak. Release when done.")
        print("Press Ctrl+C after testing to end.")
        
        try:
            while True:
                # Record audio
                start_time = time.time()
                audio = self.recorder.record()
                recording_time = time.time() - start_time
                
                # Convert speech to text
                transcription_start = time.time()
                result = self.stt(audio)
                transcription_time = time.time() - transcription_start
                
                # Print results
                print("\nRecorded audio length: {:.2f}s".format(recording_time))
                print("Transcription time: {:.2f}s".format(transcription_time))
                print("Transcribed text:", result)
                
        except KeyboardInterrupt:
            print("\nTest ended by user")
        finally:
            self.recorder.cleanup()

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'recorder'):
            self.recorder.cleanup()


if __name__ == '__main__':
    unittest.main()
