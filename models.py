# models.py
# Handles Hugging Face model loading and inference.
# NOTE: This code uses `transformers` pipelines. The user must install dependencies:
# pip install transformers torch pillow

from transformers import pipeline
import functools

class ModelManager:
    def __init__(self):
        # we'll lazy-load models on first use
        self._text_gen = None
        self._image_clf = None

    def _ensure_text_model(self):
        if self._text_gen is None:
            # Using a small model (gpt2) to keep downloads moderate.
            self._text_gen = pipeline('text-generation', model='gpt2')
        return self._text_gen

    def _ensure_image_model(self):
        if self._image_clf is None:
            # ViT image classification (small-ish)
            self._image_clf = pipeline('image-classification', model='google/vit-base-patch16-224')
        return self._image_clf

    def generate_text(self, prompt, max_length=100):
        gen = self._ensure_text_model()
        out = gen(prompt, max_length=max_length, num_return_sequences=1)
        # pipeline returns a list of dicts with 'generated_text'
        return out[0].get('generated_text', str(out))

    def classify_image(self, image_bytes_io):
        clf = self._ensure_image_model()
        out = clf(image_bytes_io)
        # Format output nicely
        lines = []
        for item in out:
            label = item.get('label', '')
            score = item.get('score', 0.0)
            lines.append(f"{label}: {score:.4f}")
        return '\n'.join(lines)

    def get_models_info(self):
        info = []
        info.append('text-generation (gpt2) - small autoregressive model for text generation (uses transformers pipeline).')
        info.append('image-classification (google/vit-base-patch16-224) - Vision Transformer for image classification.')
        info.append('\nNotes:\n- These models are fetched from Hugging Face on first use.\n- You need internet when running the program for the first time to download model weights.')
        return '\n\n'.join(info)
