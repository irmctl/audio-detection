"""Frozen feature extractors for multi-model audio deepfake probing."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FeatureExtractor(Protocol):
    name: str

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray: ...


class WhisperExtractor:
    name = "whisper"

    def __init__(self, model_id: str = "openai/whisper-base", device: str | None = None):
        from transformers import WhisperModel, WhisperProcessor

        self.device = resolve_device(device)
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        feats = inputs.input_features.to(self.device)
        hidden = self.model.encoder(feats).last_hidden_state.squeeze(0)
        return hidden.mean(dim=0).float().cpu().numpy()


class Wav2Vec2Extractor:
    name = "wav2vec2"

    def __init__(self, model_id: str = "facebook/wav2vec2-base", device: str | None = None):
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        self.device = resolve_device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        values = inputs.input_values.to(self.device)
        hidden = self.model(values).last_hidden_state.squeeze(0)
        return hidden.mean(dim=0).float().cpu().numpy()


class WavLMExtractor:
    name = "wavlm"

    def __init__(self, model_id: str = "microsoft/wavlm-base", device: str | None = None):
        from transformers import WavLMModel, AutoFeatureExtractor

        self.device = resolve_device(device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        values = inputs.input_values.to(self.device)
        hidden = self.model(values).last_hidden_state.squeeze(0)
        return hidden.mean(dim=0).float().cpu().numpy()


class AASISTExtractor:
    """Use ONNX AASIST; features = [bonafide_logit, spoof_logit] (+ optional score)."""

    name = "aasist"

    def __init__(self, device: str | None = None):
        import onnxruntime as ort

        onnx_path = hf_hub_download(
            "SpeechAntiSpoofingBenchmarks/AASIST",
            "aasist.onnx",
        )
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device != "cpu"
            else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.target_len = 64600

    def _pad(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32)
        if x.shape[0] >= self.target_len:
            return x[: self.target_len]
        out = np.zeros(self.target_len, dtype=np.float32)
        out[: x.shape[0]] = x
        return out

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        x = self._pad(audio)
        # try common input shapes
        for arr in (
            x[None, None, :],
            x[None, :],
            x,
        ):
            try:
                outs = self.session.run(None, {self.input_name: arr.astype(np.float32)})
                vec = np.concatenate([np.asarray(o).reshape(-1) for o in outs]).astype(np.float32)
                if vec.size == 1:
                    # single bona-fide score -> expand
                    s = float(vec[0])
                    return np.array([s, -s], dtype=np.float32)
                return vec
            except Exception:
                continue
        raise RuntimeError("AASIST ONNX forward failed for all input shapes")


class Qwen2AudioExtractor:
    """Qwen2-Audio audio tower mean pooling (heavy)."""

    name = "qwen2_audio"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-Audio-7B",
        device: str | None = None,
    ):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        local = Path("data/models/qwen2-audio-7b")
        if (local / "config.json").exists() and any(local.glob("model-*.safetensors")):
            model_id = str(local.resolve())
            print(f"Using local Qwen path: {model_id}")

        self.device = resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.device.type == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs)
        elif self.device.type == "mps":
            # MPS: no device_map; float16 is usually fine on Apple Silicon.
            kwargs["torch_dtype"] = torch.float16
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs)
            self.model.to(self.device)
        else:
            kwargs["torch_dtype"] = torch.float32
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs)
            self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        # Truncate to 30s max for memory
        max_len = sr * 30
        if audio.shape[0] > max_len:
            audio = audio[:max_len]
        inputs = self.processor(
            text=["<|audio_bos|><|AUDIO|><|audio_eos|>Detect if fake."],
            audio=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        # Prefer audio tower hidden states if available
        audio_tower = getattr(self.model, "audio_tower", None) or getattr(
            self.model, "model", None
        )
        if audio_tower is not None and hasattr(self.model, "audio_tower"):
            # Qwen2-Audio: use feature extractor path via get_audio_features if present
            if hasattr(self.model, "get_audio_features"):
                feats = self.model.get_audio_features(
                    inputs.get("input_features")
                    if "input_features" in inputs
                    else inputs.get("audio_values")
                )
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                h = feats.squeeze(0)
                return h.mean(dim=0).float().cpu().numpy()

        # Fallback: run encoder-ish forward and pool last hidden if exposed
        out = self.model(**inputs, output_hidden_states=True)
        if hasattr(out, "hidden_states") and out.hidden_states:
            h = out.hidden_states[-1].squeeze(0)
            return h.mean(dim=0).float().cpu().numpy()
        # last resort: use logits mean
        logits = out.logits.squeeze(0)
        return logits.mean(dim=0).float().cpu().numpy()


def build_extractor(name: str, device: str | None = None):
    name = name.lower().replace("-", "_")
    if name in {"whisper"}:
        return WhisperExtractor(device=device)
    if name in {"wav2vec", "wav2vec2"}:
        return Wav2Vec2Extractor(device=device)
    if name in {"wavlm"}:
        return WavLMExtractor(device=device)
    if name in {"aasist"}:
        return AASISTExtractor(device=device)
    if name in {"qwen2_audio", "qwen2audio", "qwen"}:
        return Qwen2AudioExtractor(device=device)
    raise ValueError(f"Unknown model: {name}")
