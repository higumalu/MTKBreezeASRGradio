"""Gradio app for chunked transcription using MediaTek-Research/Breeze-ASR-25."""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import gradio as gr
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


TARGET_SAMPLE_RATE = 16_000
CHUNK_SECONDS = 10.0
CHUNK_OVERLAP_SECONDS = 2.0
MODEL_ID = "MediaTek-Research/Breeze-ASR-25"


@dataclass
class ChunkResult:
    index: int
    start: float
    end: float
    text: str
    confidence: float
    audio_path: Path


@dataclass
class SessionArtifacts:
    session_id: str
    audio_path: Path
    chunk_dir: Path
    subtitles_dir: Path
    chunk_results: List[ChunkResult] = field(default_factory=list)


class ArtifactManager:
    """Handles persistence of audio, chunk, and subtitle files."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path(os.environ.get("ASR_APP_CACHE", tempfile.gettempdir())) / "breeze_asr"
        self.audio_cache = self.base_dir / "audio"
        self.subtitles_cache = self.base_dir / "subtitles"
        self.audio_cache.mkdir(parents=True, exist_ok=True)
        self.subtitles_cache.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionArtifacts] = {}

    # region public API
    def register_session(self) -> SessionArtifacts:
        session_id = str(uuid.uuid4())
        chunk_dir = self.audio_cache / session_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        session = SessionArtifacts(
            session_id=session_id,
            audio_path=self.audio_cache / f"{session_id}.wav",
            chunk_dir=chunk_dir,
            subtitles_dir=self.subtitles_cache,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def record_chunk_result(self, session: SessionArtifacts, result: ChunkResult) -> None:
        session.chunk_results.append(result)

    def get_session_artifacts(self) -> List[SessionArtifacts]:
        with self._lock:
            return list(self._sessions.values())

    def cleanup(self) -> str:
        """Remove cached files from disk and reset state."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self.audio_cache.mkdir(parents=True, exist_ok=True)
        self.subtitles_cache.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._sessions.clear()
        return f"已清除暫存資料：{self.base_dir}"  # Chinese message per request
    # endregion


def ensure_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        return audio.unsqueeze(0)
    if audio.shape[0] == 1:
        return audio
    return torch.mean(audio, dim=0, keepdim=True)


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    return torchaudio.functional.resample(audio, orig_sr, target_sr)


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    torchaudio.save(str(path), audio, sample_rate)


def chunk_audio(
    audio: torch.Tensor,
    sample_rate: int,
    chunk_seconds: float,
    destination: Path,
    overlap_seconds: float = 0.0,
) -> Iterable[Tuple[int, float, float, Path, torch.Tensor]]:
    chunk_samples = int(chunk_seconds * sample_rate)
    total_samples = audio.shape[-1]
    total_duration = total_samples / sample_rate
    overlap_samples = int(overlap_seconds * sample_rate)
    step_samples = max(chunk_samples - overlap_samples, 1)
    idx = 0
    start_sample = 0
    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = audio[:, start_sample:end_sample]
        if chunk.shape[-1] == 0:
            break
        if chunk.shape[-1] < chunk_samples:
            padding = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        start_sec = start_sample / sample_rate
        end_sec = min(start_sec + chunk_seconds, total_duration)
        chunk_path = destination / f"chunk_{idx:04d}.wav"
        torchaudio.save(str(chunk_path), chunk, sample_rate)
        yield idx, start_sec, end_sec, chunk_path, chunk
        idx += 1
        start_sample += step_samples


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((float(seconds) - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(session: SessionArtifacts) -> Path:
    srt_path = session.subtitles_dir / f"{session.session_id}.srt"
    lines: List[str] = []
    for idx, chunk in enumerate(session.chunk_results, 1):
        lines.append(str(idx))
        start = format_timestamp(chunk.start)
        end = format_timestamp(chunk.end)
        lines.append(f"{start} --> {end}")
        confidence = f"(confidence: {chunk.confidence:.2f})" if chunk.confidence else ""
        lines.append(f"{chunk.text} {confidence}".strip())
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def generate_txt(session: SessionArtifacts) -> Path:
    txt_path = session.subtitles_dir / f"{session.session_id}.txt"
    lines = []
    for chunk in session.chunk_results:
        timestamp = f"[{format_timestamp(chunk.start)} - {format_timestamp(chunk.end)}]"
        lines.append(f"{timestamp} ({chunk.confidence:.2f}) {chunk.text}".strip())
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return txt_path


class BreezeASR:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        max_target_positions = getattr(self.model.config, "max_target_positions", None)
        if max_target_positions:
            # Reserve space for decoder start tokens that Whisper prepends automatically.
            safe_max_new = max(max_target_positions - 4, 1)
            self.generation_kwargs = dict(max_new_tokens=safe_max_new)
        else:
            self.generation_kwargs = dict()

    def transcribe_chunk(self, audio: torch.Tensor, sample_rate: int) -> Tuple[str, float]:
        audio = ensure_mono(audio)
        inputs = self.processor(
            audio.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.get("input_features")
        if input_features is None:
            raise ValueError("Processor did not return input_features")
        input_features = input_features.to(self.device, dtype=self.model.dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                **self.generation_kwargs,
            )
        text = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
        confidence = self._compute_confidence(generated)
        return text, confidence

    @staticmethod
    def _compute_confidence(generated) -> float:
        scores = getattr(generated, "scores", None)
        sequences = getattr(generated, "sequences", None)
        if not scores or sequences is None:
            return 0.0
        sequence = sequences[0]
        if len(scores) == 0:
            return 0.0
        # Align scores with generated tokens (skip BOS token)
        token_ids = sequence[-len(scores):]
        log_probs: List[float] = []
        for token_id, score in zip(token_ids, scores):
            log_probabilities = torch.nn.functional.log_softmax(score, dim=-1)
            flattened = log_probabilities.reshape(-1)
            if token_id < flattened.numel():
                log_prob = flattened[token_id].item()
            else:
                # Forced decoder tokens can result in singleton score tensors; treat them as certainty.
                log_prob = 0.0
            log_probs.append(log_prob)
        if not log_probs:
            return 0.0
        avg_log_prob = sum(log_probs) / len(log_probs)
        return float(math.exp(avg_log_prob))


artifact_manager = ArtifactManager()
asr_model: Optional[BreezeASR] = None


def ensure_model_loaded() -> BreezeASR:
    global asr_model
    if asr_model is None:
        asr_model = BreezeASR()
    return asr_model


def prepare_audio_file(audio_path: str, session: SessionArtifacts) -> Tuple[torch.Tensor, int]:
    waveform, orig_sr = torchaudio.load(audio_path)
    waveform = ensure_mono(waveform)
    waveform = resample_audio(waveform, orig_sr, TARGET_SAMPLE_RATE)
    save_audio(session.audio_path, waveform, TARGET_SAMPLE_RATE)
    return waveform, TARGET_SAMPLE_RATE


def format_chunk_table(chunks: List[ChunkResult]) -> List[List[str]]:
    table: List[List[str]] = []
    for chunk in chunks:
        table.append([
            format_timestamp(chunk.start).replace(",", ":"),
            format_timestamp(chunk.end).replace(",", ":"),
            f"{chunk.confidence:.2f}",
            chunk.text,
        ])
    return table


def transcribe_audio(audio_path: Optional[str]) -> Generator:
    if not audio_path:
        yield gr.update(value=""), gr.update(value=[]), None, None, "請先提供音訊檔案"
        return

    session = artifact_manager.register_session()
    waveform, sample_rate = prepare_audio_file(audio_path, session)
    model = ensure_model_loaded()

    transcript_lines: List[str] = []
    for idx, start, end, chunk_path, chunk_tensor in chunk_audio(
        waveform,
        sample_rate,
        CHUNK_SECONDS,
        session.chunk_dir,
        overlap_seconds=CHUNK_OVERLAP_SECONDS,
    ):
        text, confidence = model.transcribe_chunk(chunk_tensor, sample_rate)
        result = ChunkResult(
            index=idx,
            start=start,
            end=end,
            text=text,
            confidence=confidence,
            audio_path=chunk_path,
        )
        artifact_manager.record_chunk_result(session, result)
        transcript_lines.append(f"{format_timestamp(start)} ➜ {text}")
        table = format_chunk_table(session.chunk_results)
        status = f"處理中：第 {idx + 1} 段 (信心 {confidence:.2f})"
        yield "\n".join(transcript_lines), gr.update(value=table), None, None, status

    srt_path = generate_srt(session)
    txt_path = generate_txt(session)
    final_status = f"完成字幕產生，共 {len(session.chunk_results)} 段"
    table = format_chunk_table(session.chunk_results)
    yield "\n".join(transcript_lines), gr.update(value=table), str(srt_path), str(txt_path), final_status


def cleanup_cache() -> str:
    return artifact_manager.cleanup()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Breeze ASR Captioner") as app:
        gr.Markdown("## MediaTek Breeze ASR 即時字幕工具")
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="錄音或上傳音訊",
        )
        with gr.Row():
            transcribe_button = gr.Button("開始轉錄", variant="primary")
            cleanup_button = gr.Button("清理暫存")
        transcript_box = gr.Textbox(label="逐段字幕", value="", lines=10)
        confidence_table = gr.Dataframe(
            headers=["開始", "結束", "信心", "內容"],
            label="逐段信心",
            datatype=["str", "str", "str", "str"],
            interactive=False,
        )
        with gr.Row():
            srt_file = gr.File(label="下載 SRT", interactive=False)
            txt_file = gr.File(label="下載 TXT", interactive=False)
        status_box = gr.Markdown("等待音訊輸入...")

        transcribe_button.click(
            fn=transcribe_audio,
            inputs=audio_input,
            outputs=[transcript_box, confidence_table, srt_file, txt_file, status_box],
        )
        cleanup_button.click(fn=cleanup_cache, outputs=status_box)
        app.queue()
    return app


def main() -> None:
    interface = build_interface()
    interface.launch(share=False)


if __name__ == "__main__":
    main()
