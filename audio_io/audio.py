import threading
import queue
import numpy as np
import jax.numpy as jnp
import pyaudio
from pynput import keyboard

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        """Initialize the audio recorder.
        
        Args:
            sample_rate (int): Sample rate in Hz (default: 16000 for Whisper)
            channels (int): Number of audio channels (default: 1 for mono)
            chunk_size (int): Number of frames per buffer
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Set up keyboard listener
        self.listener = None  # Initialize as None
        
    def _on_press(self, key):
        """Callback for key press events."""
        if key == keyboard.Key.space and not self.is_recording:
            self.is_recording = True
            self.audio_queue.queue.clear()  # Clear any old audio
            self._start_recording()
            
    def _on_release(self, key):
        """Callback for key release events."""
        if key == keyboard.Key.space:
            self.is_recording = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream to process incoming audio data."""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _start_recording(self):
        """Start the audio stream."""
        print("Starting audio stream...")
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
    
    def record(self):
        """Record audio while spacebar is held.
        
        Returns:
            jnp.ndarray: Recorded audio as a JAX numpy array
        """
        print("Hold spacebar to record audio...")
        
        # Create and start a new listener only if one isn't already running
        if self.listener is None or not self.listener.is_alive():
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self.listener.start()
        
        try:
            while True:
                # Wait for spacebar press and recording to complete
                if not self.is_recording and not self.audio_queue.empty():
                    print("Audio queue is not empty, collecting audio chunks...")
                    # Collect all audio chunks
                    audio_chunks = []
                    while not self.audio_queue.empty():
                        audio_chunks.append(self.audio_queue.get())
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(b''.join(audio_chunks), dtype=np.float32)
                    return jnp.array(audio_data)
                
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if self.listener is not None:
            self.listener.stop()
        self.audio.terminate()
