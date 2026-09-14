import torch
import torchaudio
import whisperx
from typing import List
import numpy as np

class WhisperXSegmenter:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "large-v2",
        compute_type: str = "float16",
        language: str = "en",
        sample_rate: int = 16000,
    ) -> None:
        self.device = device
        self.model = whisperx.load_model(model_name, device, compute_type=compute_type)
        self.align_model, self.metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )
        self.sample_rate = sample_rate

    def segment_audio(self, audio_path: str, language: str = "en") -> List[np.ndarray]:
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, language=language)
        # language = result["language"]
        aligned = whisperx.align(
            result["segments"],
            self.align_model,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        segments = aligned["segments"]

        sliced_segments: List[np.ndarray] = []
        total_samples = len(audio)
        for seg in segments:
            start = seg.get("start")
            end = seg.get("end")
            print(f"Start: {start}, End: {end}, Text: {seg.get('text', '')}")
            if start is None or end is None:
                continue
            start_idx = int(max(0, round(float(start) * self.sample_rate)))
            end_idx = int(min(total_samples, round(float(end) * self.sample_rate)))
            if end_idx <= start_idx:
                continue
            sliced_segments.append(audio[start_idx:end_idx])

        # fallback: no valid ASR segmentation, use whole doc audio
        if not sliced_segments:
            print("No valid ASR segmentation, using whole doc audio")
            sliced_segments.append(audio)
        return sliced_segments
