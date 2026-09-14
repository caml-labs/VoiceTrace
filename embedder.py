import abc
import numpy as np
import torch
import torchaudio


class SpeakerEmbedder(abc.ABC):
    """Abstract base class for speaker embedding models."""

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb) + 1e-12
        return emb / norm

    @abc.abstractmethod
    def embed_waveform(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        ...

    def embed_file(self, audio_path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(audio_path)
        return self.embed_waveform(waveform, sr)

    def embed_mono_numpy(self, audio_mono: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        wav = torch.from_numpy(audio_mono).float().unsqueeze(0)
        return self.embed_waveform(wav, sample_rate)


class ECAPASpeakerEmbedder(SpeakerEmbedder):
    """Speaker embedder backed by the SpeechBrain ECAPA-TDNN model."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": device},
        )

    def embed_waveform(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        try:
            # ECAPA expects (channel, time)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            # Resample to 16 kHz if needed (ECAPA trained at 16 kHz)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
            embeddings = self.classifier.encode_batch(waveform)
            emb = embeddings.squeeze().cpu().numpy()
            return self._normalize(emb)
        except Exception as e:
            print(f"Error embedding waveform: waveform shape {waveform.shape} sample rate {sample_rate} error {e}")
            return np.zeros(192)