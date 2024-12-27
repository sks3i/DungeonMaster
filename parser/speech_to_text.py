import jax
import jax.numpy as jnp

from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration


class SpeechToText:
    def __init__(self, 
                 model_name: str = "openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = FlaxWhisperForConditionalGeneration.from_pretrained(model_name)

        # Move parameters to GPU if available
        if jax.default_backend() == "gpu":
            self.model.params = jax.device_put(self.model.params, jax.devices("gpu")[0])

    def __call__(self, audio: jnp.ndarray) -> str:
        # Ensure audio is 1D and float32
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono if needed
        audio = audio.astype(jnp.float32)
        
        # Process the audio
        inputs = self.processor(
            audio, 
            return_tensors="jax", 
            sampling_rate=16000,
            padding="max_length",
            max_length=480000
        )
        
        # Generate and ensure outputs are integers
        outputs = self.model.generate(**inputs).sequences
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

