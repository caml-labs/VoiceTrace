import whisperx
from typing import List, Tuple
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

    def segment_audio_with_text(
        self, audio_path: str, language: str = "en"
    ) -> Tuple[List[np.ndarray], List[str]]:
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, language=language)
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
        segment_texts: List[str] = []
        total_samples = len(audio)
        for seg in segments:
            start = seg.get("start")
            end = seg.get("end")
            text = str(seg.get("text", "")).strip()
            print(f"Start: {start}, End: {end}, Text: {text}")
            if start is None or end is None:
                continue
            start_idx = int(max(0, round(float(start) * self.sample_rate)))
            end_idx = int(min(total_samples, round(float(end) * self.sample_rate)))
            if end_idx <= start_idx:
                continue
            sliced_segments.append(audio[start_idx:end_idx])
            segment_texts.append(text)

        # fallback: no valid ASR segmentation, use whole doc audio
        if not sliced_segments:
            print("No valid ASR segmentation, using whole doc audio")
            sliced_segments.append(audio)
            segment_texts.append(
                " ".join(
                    str(seg.get("text", "")).strip()
                    for seg in result.get("segments", [])
                    if seg.get("text")
                )
            )
        return sliced_segments, segment_texts

    def segment_audio(self, audio_path: str, language: str = "en") -> List[np.ndarray]:
        sliced_segments, _ = self.segment_audio_with_text(audio_path, language=language)
        return sliced_segments
