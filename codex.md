# Colab + Gradio Breeze ASR App Plan

## 1. Environment Setup (Colab)
- Install core libs: `pip install gradio soundfile torchaudio transformers accelerate datasets sentencepiece`.
- Mount Google Drive (optional) if long-term storage of audio/subtitles is needed.
- Create env variables for Hugging Face access token and call `huggingface-cli login` to pull `MediaTek-Research/Breeze-ASR-25`.

## 2. Model Initialization
- Use `AutoProcessor` + `AutoModelForSpeechSeq2Seq` from `transformers` to load `MediaTek-Research/Breeze-ASR-25`.
- Detect GPU availability (`torch.cuda.is_available()`), move model to GPU, enable `torch.float16` if VRAM allows.
- Wrap inference in `transcribe_chunk(audio_chunk)` to accept 16 kHz mono torch tensors and return text + confidence.

## 3. Audio Ingestion and Storage
- Gradio `gr.Audio(sources=["microphone","upload"], type="filepath")` to collect inputs.
- After recording/upload, immediately convert audio to 16 kHz mono WAV via `torchaudio.load/resample` or `soundfile` and save as `/content/audio_cache/<uuid>.wav`.
- Keep metadata (duration, sample rate, file path) in a lightweight registry (Python dict or SQLite if needed).

## 4. Chunking Pipeline (10 s patches)
- Implement `chunk_audio(file_path, chunk_len=10.0)` that yields `(chunk_index, start_sec, end_sec, tensor)`.
- Persist chunk WAVs into `/content/audio_cache/<uuid>/chunk_<idx>.wav` so users can inspect raw clips if desired.
- Use background thread or `asyncio` task to process chunks as they become available, updating UI incrementally.

## 5. ASR + Confidence Scoring
- For each chunk, run `transcribe_chunk` and collect decoder output probabilities.
- Derive per-chunk confidence (e.g., average log-prob or use `processor.tokenizer.decode` with scores) and store with the transcript.
- Maintain running transcript list: `[{"start": start, "end": end, "text": text, "confidence": score}]`.

## 6. Live Subtitles in Gradio
- Display rolling transcript using `gr.Textbox` or `gr.HighlightedText`; update via `yield gr.update(value=formatted_text)`.
- Show per-chunk confidence alongside timestamps (e.g., formatted string `00:00-00:10 | 0.91 | text`).
- Optionally add a progress bar tied to processed chunk count vs total.

## 7. Subtitle Export (.srt / .txt)
- Implement utility `generate_srt(chunks)` that formats timestamps into `HH:MM:SS,mmm` and writes `/content/subtitles/<uuid>.srt`.
- Create `generate_txt(chunks)` for plain text transcript with timestamps and confidences.
- Expose `gr.File` components or download buttons that appear once transcription is complete. Bundle both files in a ZIP if desired.

## 8. Cleanup Controls
- Track all generated paths in a session cache.
- Provide `gr.Button("清理暫存")`; on click, remove cached audio/chunk/subtitle files with `shutil.rmtree` and clear in-memory registries.
- Confirm cleanup status in UI (`gr.Markdown` update) so users know storage has been reclaimed.

## 9. Testing & Validation
- Include a sample audio clip in the notebook for quick sanity checks.
- Add utility cell to measure latency per chunk and overall throughput (CPU vs GPU).
- Log errors and handle edge cases (shorter than 10 s audio, empty recordings, missing token).

## 10. Notebook Flow
1. Setup cell (installs, auth, directory creation).
2. Model load cell (with progress + VRAM note).
3. Helper functions cell (audio utils, chunking, subtitle formatting, cleanup).
4. ASR worker cell (transcription loop with confidence extraction).
5. Gradio UI launch cell (defines interface, binds events, `app.launch(share=True)`).
6. Optional testing cell (runs pipeline on bundled sample audio).

## 11. Future Enhancements
- Support streaming microphone chunks (WebRTC) for lower-latency captioning.
- Add multilingual or diarization options if future models are plugged in.
- Persist transcripts to Google Drive with metadata for later summarization.
