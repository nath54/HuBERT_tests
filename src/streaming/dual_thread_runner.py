"""
Dual-Threaded Streaming Speech Recognition Engine.
Implements asynchronous parallel inference:
- Thread 1 (Acoustic Worker): Consumes audio chunks, executes Model_A2P (Audio -> Phonemes)
  with sliding acoustic context and energy-based VAD, emits recognized phonemes in real-time,
  and pushes word-segmented phonemes across space boundaries (token 8) into a thread-safe queue.
- Thread 2 (Decoder Worker): Consumes phoneme-sliced words from the queue, executes
  Model_PE (Phonemes -> Word Latents z_word) and Model_WD (Word Latents -> Orthographic Text)
  with Trie dictionary constraints, emitting real-time word tokens with minimal latency.
"""

# Import Modules
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
import time
import torch
import torch.nn.functional as F
from torch import Tensor

from ..model1_a2p import Model_A2P
from ..model2_pe import Model_PE
from ..model3_wd import Model_WD
from ..tokenizers.phoneme_tokenizer import PhonemeTokenizer
from .stream_buffer import AudioCircularBuffer, PhonemeWordQueue


class DualThreadStreamingASR:
    """
    Dual-Threaded Asynchronous Streaming Speech-to-Text Pipeline.
    Decouples heavy acoustic processing (Model 1) from linguistic word synthesis (Models 2 & 3).
    """

    def __init__(
        self,
        model_a2p: Model_A2P,
        model_pe: Model_PE,
        model_wd: Model_WD,
        chunk_duration_sec: float = 0.5,
        context_duration_sec: float = 1.0,
        sample_rate: int = 16000,
        word_callback: Optional[Callable[[str], None]] = None,
        phoneme_callback: Optional[Callable[[str], None]] = None,
        trie: Optional[Any] = None,
        beam_width: int = 1,
        vad_threshold: float = 0.005,
        min_phonemes_per_word: int = 1,
        max_phonemes_per_word: int = 32,
        space_id: int = 8,
        blank_id: int = 1,
        pad_id: int = 0,
        blank_penalty: float = 2.6,
        space_bonus: float = 0.35,
        allowed_phoneme_range: Optional[Tuple[int, int]] = (1, 111),
        max_queue_size: int = 500,
    ) -> None:
        """
        Initialize the DualThreadStreamingASR pipeline.

        Args:
            model_a2p (Model_A2P): Part 1 Acoustic-to-Phoneme model.
            model_pe (Model_PE): Part 2 Phoneme-to-Embedding encoder.
            model_wd (Model_WD): Part 3 Word Decoder.
            chunk_duration_sec (float): Duration in seconds of audio chunks processed per acoustic step (default: 0.5s).
            context_duration_sec (float): Duration in seconds of past audio context kept for Conformer receptive field (default: 1.0s).
            sample_rate (int): Input audio sampling rate (default: 16000 Hz).
            word_callback (Optional[Callable[[str], None]]): Callback invoked when an orthographic word is decoded.
            phoneme_callback (Optional[Callable[[str], None]]): Callback invoked when phoneme tokens or boundaries are recognized.
            trie (Optional[Any]): Vocabulary Trie prefix tree for dictionary-constrained decoding.
            beam_width (int): Beam width for character decoder (1 = greedy rollout).
            vad_threshold (float): Baseline RMS energy threshold for Voice Activity Detection silence gating.
            min_phonemes_per_word (int): Minimum phonemes required to emit a word (default: 1).
            max_phonemes_per_word (int): Maximum phoneme length passed to Model 2 (default: 32).
            space_id (int): Phoneme token ID representing acoustic word boundaries (default: 8).
            blank_id (int): CTC blank token ID (default: 1).
            pad_id (int): Batch padding token ID (default: 0).
            blank_penalty (float): CTC blank penalty for acoustic decoding.
            space_bonus (float): CTC space token bonus for acoustic decoding.
            allowed_phoneme_range (Optional[Tuple[int, int]]): Valid phoneme token range [min, max] (default: (1, 111)).
            max_queue_size (int): Max queued words before blocking acoustic thread (default: 500).
        """
        # Model components
        self.model_a2p: Model_A2P = model_a2p
        self.model_pe: Model_PE = model_pe
        self.model_wd: Model_WD = model_wd

        # Tokenizers & Vocabulary Trie
        self.phoneme_tokenizer: PhonemeTokenizer = PhonemeTokenizer()
        self.trie: Optional[Any] = trie
        self.beam_width: int = max(1, beam_width)

        # Audio stream settings
        self.sample_rate: int = sample_rate
        self.chunk_size_samples: int = int(chunk_duration_sec * sample_rate)
        self.context_size_samples: int = int(context_duration_sec * sample_rate)
        self.vad_threshold: float = vad_threshold

        # Phoneme segmentation parametrization
        self.min_phonemes_per_word: int = min_phonemes_per_word
        self.max_phonemes_per_word: int = max_phonemes_per_word
        self.space_id: int = space_id
        self.blank_id: int = blank_id
        self.pad_id: int = pad_id
        self.blank_penalty: float = blank_penalty
        self.space_bonus: float = space_bonus
        self.allowed_phoneme_range: Optional[Tuple[int, int]] = allowed_phoneme_range

        # Thread-safe inter-thread communication buffers
        self.audio_buffer: AudioCircularBuffer = AudioCircularBuffer(capacity_samples=sample_rate * 60)
        self.word_queue: PhonemeWordQueue = PhonemeWordQueue(maxsize=max_queue_size)

        # Callbacks for real-time live streaming
        self.word_callback: Optional[Callable[[str], None]] = word_callback
        self.phoneme_callback: Optional[Callable[[str], None]] = phoneme_callback

        # Internal state
        self._running: bool = False
        self._stop_requested: bool = False
        self._acoustic_thread: Optional[threading.Thread] = None
        self._decoder_thread: Optional[threading.Thread] = None

        # Performance profiling statistics
        self._chunks_processed: int = 0
        self._total_acoustic_time: float = 0.0
        self._words_decoded: int = 0
        self._total_decoder_time: float = 0.0

        # Accumulated decoded transcription tokens
        self._decoded_words: List[str] = []
        self._words_lock: threading.Lock = threading.Lock()

    def calibrate_noise_floor(self, calibration_samples: Tensor) -> float:
        """
        Calibrates the ambient noise floor and adjusts the dynamic VAD threshold.

        Args:
            calibration_samples (Tensor): 1D float tensor of ambient silence samples.

        Returns:
            float: Calibrated VAD threshold.
        """
        if calibration_samples.numel() == 0:
            return self.vad_threshold

        clean_samples = calibration_samples - torch.mean(calibration_samples)
        noise_rms = float(torch.sqrt(torch.mean(clean_samples ** 2)).item())
        # Set dynamic threshold at 2.8x ambient noise RMS with safe baseline of 0.005
        self.vad_threshold = max(0.005, noise_rms * 2.8)
        return self.vad_threshold

    def start(self) -> None:
        """
        Starts the dual-threaded pipeline. Launches both acoustic and decoder daemon threads.
        """
        if self._running:
            return

        self._running = True
        self._stop_requested = False
        with self._words_lock:
            self._decoded_words.clear()

        # Thread 1: Acoustic Model 1 Thread
        self._acoustic_thread = threading.Thread(
            target=self._acoustic_thread_loop,
            name="Acoustic-Worker-Thread-1",
            daemon=True,
        )

        # Thread 2: Decoder Models 2 & 3 Thread
        self._decoder_thread = threading.Thread(
            target=self._decoder_thread_loop,
            name="Decoder-Worker-Thread-2",
            daemon=True,
        )

        self._acoustic_thread.start()
        self._decoder_thread.start()

    def push_audio(self, audio_chunk: Tensor) -> int:
        """
        Feeds raw 16kHz audio samples into the stream buffer.

        Args:
            audio_chunk (Tensor): Float audio tensor [T] or [1, T].

        Returns:
            int: Number of samples accepted into circular buffer.
        """
        return self.audio_buffer.write(audio_chunk)

    def finish_stream(self, timeout: float = 10.0) -> str:
        """
        Signals end of audio stream, waits for both threads to finalize and decode remaining words.

        Args:
            timeout (float): Maximum wait time in seconds for worker threads to complete.

        Returns:
            str: Complete final transcribed orthographic sentence.
        """
        # Request acoustic thread to flush remaining buffer and terminate
        self._stop_requested = True

        if self._acoustic_thread is not None and self._acoustic_thread.is_alive():
            self._acoustic_thread.join(timeout=timeout)

        # Signal EOS to decoder thread
        self.word_queue.push_end_of_stream()

        if self._decoder_thread is not None and self._decoder_thread.is_alive():
            self._decoder_thread.join(timeout=timeout)

        self._running = False

        # Return full accumulated sentence
        with self._words_lock:
            return " ".join(self._decoded_words).strip()

    def get_partial_transcription(self) -> str:
        """
        Returns the current live partial transcription.

        Returns:
            str: Partial text decoded up to the present moment.
        """
        with self._words_lock:
            return " ".join(self._decoded_words).strip()

    def is_running(self) -> bool:
        """
        Returns True if streaming worker threads are actively running.

        Returns:
            bool: Running status.
        """
        return self._running

    # -------------------------------------------------------------------------
    # Worker Thread 1: Model 1 Acoustic Processing with Sliding Context
    # -------------------------------------------------------------------------
    def _acoustic_thread_loop(self) -> None:
        """
        Acoustic worker thread executing Model_A2P on audio stream chunks.
        Applies past sliding window context for Conformer receptive field,
        segments phonemes at space boundaries, and pushes to word queue.
        """
        self.model_a2p.eval()
        dev_a2p: torch.device = self.model_a2p.get_device()

        # Acoustic past context buffer: Tensor [T_context]
        past_context: Tensor = torch.zeros(0, dtype=torch.float32)

        # Accumulated intra-word phoneme tokens waiting for space delimiter: List[int]
        current_word_phonemes: List[int] = []

        # CTC state: last emitted token ID to prevent duplicate collapse across chunk boundaries
        last_ctc_token_id: int = self.blank_id

        while self._running:
            # Check available audio in buffer
            available: int = self.audio_buffer.get_available_samples()

            # If stopping and buffer has remaining samples, process all remaining; otherwise wait for chunk
            if self._stop_requested:
                if available == 0:
                    break
                read_count: int = available
            else:
                if available < self.chunk_size_samples:
                    time.sleep(0.01)
                    continue
                read_count = self.chunk_size_samples

            # Read chunk: Tensor [read_count], Role: current streaming audio frame
            audio_raw: Optional[Tensor] = self.audio_buffer.read(num_samples=read_count, block=False)
            if audio_raw is None:
                continue

            # Model 1 requires at least 400 samples (25ms) for 1 CNN window
            if audio_raw.shape[0] < 400:
                continue

            # Remove DC bias
            audio_clean = audio_raw - torch.mean(audio_raw)

            # --- Energy-Based Voice Activity Detection (VAD) / Silence Gate ---
            # Suppresses false phoneme hallucinations during microphone silence/pauses
            rms_energy: float = float(torch.sqrt(torch.mean(audio_clean ** 2)).item())
            if rms_energy < self.vad_threshold:
                # Silence frame: if we had an ongoing word, finalize and push to decoder
                if len(current_word_phonemes) >= self.min_phonemes_per_word:
                    clipped: List[int] = current_word_phonemes[: self.max_phonemes_per_word]
                    self.word_queue.push_word(list(clipped))
                    if self.phoneme_callback is not None:
                        try:
                            self.phoneme_callback(" | ")
                        except Exception:
                            pass
                current_word_phonemes.clear()
                last_ctc_token_id = self.blank_id
                # Update past context with silence window to preserve smooth transitions
                if self.context_size_samples > 0:
                    past_context = torch.cat([past_context, audio_clean], dim=0)[-self.context_size_samples:]
                continue

            # Assemble sliding context window: [past_context + new_audio]
            if self.context_size_samples > 0:
                window_audio: Tensor = torch.cat([past_context, audio_clean], dim=0)
                past_context = window_audio[-self.context_size_samples:]
                past_samples_count: int = window_audio.shape[0] - audio_clean.shape[0]
            else:
                window_audio = audio_clean
                past_samples_count = 0

            # Forward through Model 1 on target device
            audio_tensor: Tensor = window_audio.unsqueeze(0).to(dev_a2p)

            t_a2p_start = time.perf_counter()
            with torch.no_grad():
                a2p_out = self.model_a2p.forward(audio_tensor)
                logits = a2p_out["logits"][0]  # [T_frames, 112]

                # Slicing: only process newly arrived frames corresponding to the new chunk
                # CNN stride is 320 samples per frame
                past_frames: int = past_samples_count // 320
                new_logits = logits[past_frames:].clone()

                # Apply calibration adjustments
                if self.blank_penalty > 0.0:
                    new_logits[:, self.blank_id] -= self.blank_penalty
                if self.space_bonus != 0.0:
                    new_logits[:, self.space_id] += self.space_bonus

                # Frame-by-frame CTC argmax
                frame_preds: List[int] = new_logits.argmax(dim=-1).cpu().tolist()

            self._total_acoustic_time += (time.perf_counter() - t_a2p_start)
            self._chunks_processed += 1

            # Process frame predictions with cross-chunk CTC collapse
            for tok in frame_preds:
                if tok == last_ctc_token_id:
                    continue
                last_ctc_token_id = tok

                if tok in (self.blank_id, self.pad_id):
                    continue

                if tok == self.space_id:
                    # Space boundary: push completed word to queue
                    if len(current_word_phonemes) >= self.min_phonemes_per_word:
                        clipped_word: List[int] = current_word_phonemes[: self.max_phonemes_per_word]
                        self.word_queue.push_word(list(clipped_word))
                    current_word_phonemes.clear()

                    # Trigger phoneme callback for space boundary
                    if self.phoneme_callback is not None:
                        try:
                            self.phoneme_callback(" | ")
                        except Exception as e:
                            pass

                elif self.allowed_phoneme_range is None or (
                    self.allowed_phoneme_range[0] <= tok <= self.allowed_phoneme_range[1]
                ):
                    current_word_phonemes.append(tok)

                    # Trigger phoneme callback for recognized IPA symbol
                    if self.phoneme_callback is not None:
                        try:
                            ph_sym: str = self.phoneme_tokenizer.id_to_token.get(tok, str(tok))
                            self.phoneme_callback(f"{ph_sym} ")
                        except Exception as e:
                            pass

                    # Prevent unbounded intra-word buffers
                    if len(current_word_phonemes) >= self.max_phonemes_per_word:
                        self.word_queue.push_word(list(current_word_phonemes))
                        current_word_phonemes.clear()

        # Flush any trailing phoneme word upon stream completion
        if len(current_word_phonemes) >= self.min_phonemes_per_word:
            clipped_trailing: List[int] = current_word_phonemes[: self.max_phonemes_per_word]
            self.word_queue.push_word(list(clipped_trailing))
            current_word_phonemes.clear()

    # -------------------------------------------------------------------------
    # Worker Thread 2: Models 2 & 3 Linguistic Decoding with Trie
    # -------------------------------------------------------------------------
    def _decoder_thread_loop(self) -> None:
        """
        Decoder worker thread consuming phoneme words and executing Model_PE + Model_WD.
        Applies optional Trie dictionary constraint and beam search.
        """
        self.model_pe.eval()
        self.model_wd.eval()

        dev_pe: torch.device = self.model_pe.get_device()
        dev_wd: torch.device = self.model_wd.get_device()

        while self._running or not self.word_queue.is_empty():
            # Pop next available phoneme word sequence
            word_phonemes: Optional[List[int]] = self.word_queue.pop_word(block=True, timeout=0.05)

            if word_phonemes is None:
                if self._stop_requested and self.word_queue.is_empty():
                    break
                continue

            t_dec_start = time.perf_counter()
            # Pass word phonemes to Model 2 (PE) -> continuous word latent z_word [1, 1, 512]
            with torch.no_grad():
                # Model 2 expects List[List[int]] representing list of word slices
                z_word: Tensor = self.model_pe.forward_inference([word_phonemes]).to(dev_wd)

                # Model 3 expects z_words [B, L, 512] -> returns List[str] of decoded words
                decoded_words: List[str] = self.model_wd.forward_inference(
                    z_word,
                    trie=self.trie,
                    beam_width=self.beam_width,
                )

            self._total_decoder_time += (time.perf_counter() - t_dec_start)
            self._words_decoded += 1

            if decoded_words:
                emitted_word: str = decoded_words[0].strip()
                if emitted_word:
                    with self._words_lock:
                        self._decoded_words.append(emitted_word)

                    # Trigger user callback if provided
                    if self.word_callback is not None:
                        try:
                            self.word_callback(emitted_word)
                        except Exception as e:
                            print(f"[DualThreadStreamingASR] Callback exception: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Returns performance and latency profiling metrics.

        Returns:
            Dict[str, Any]: Metrics including processed chunks, words decoded, and average stage latency.
        """
        avg_acoustic_ms: float = (
            (self._total_acoustic_time / self._chunks_processed * 1000.0)
            if self._chunks_processed > 0
            else 0.0
        )
        avg_decoder_ms: float = (
            (self._total_decoder_time / self._words_decoded * 1000.0)
            if self._words_decoded > 0
            else 0.0
        )
        return {
            "chunks_processed": self._chunks_processed,
            "words_decoded": self._words_decoded,
            "avg_acoustic_latency_ms": round(avg_acoustic_ms, 2),
            "avg_decoder_latency_ms": round(avg_decoder_ms, 2),
            "total_latency_ms": round(avg_acoustic_ms + avg_decoder_ms, 2),
        }
