"""
Dual-Threaded Streaming Speech Recognition Engine.
Implements asynchronous parallel inference:
- Thread 1 (Acoustic Worker): Consumes audio chunks, executes Model_A2P (Audio -> Phonemes),
  and pushes word-segmented phonemes across space boundaries (token 8) into a thread-safe queue.
- Thread 2 (Decoder Worker): Consumes phoneme-sliced words from the queue, executes
  Model_PE (Phonemes -> Word Latents z_word) and Model_WD (Word Latents -> Orthographic Text),
  emitting real-time word tokens with minimal latency.
"""

# Import Modules
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
import time
import torch
from torch import Tensor

from ..model1_a2p import Model_A2P
from ..model2_pe import Model_PE
from ..model3_wd import Model_WD
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
        sample_rate: int = 16000,
        word_callback: Optional[Callable[[str], None]] = None,
        min_phonemes_per_word: int = 1,
        max_phonemes_per_word: int = 32,
        space_id: int = 8,
        blank_id: int = 0,
        allowed_phoneme_range: Optional[Tuple[int, int]] = (1, 111),
        max_queue_size: int = 500,
    ) -> None:
        """
        Initialize the DualThreadStreamingASR pipeline.

        Args:
            model_a2p (Model_A2P): Part 1 Acoustic-to-Phoneme model.
            model_pe (Model_PE): Part 2 Phoneme-to-Embedding encoder.
            model_wd (Model_WD): Part 3 Word Decoder.
            chunk_duration_sec (float): Duration in seconds of audio chunks processed per acoustic step.
            sample_rate (int): Input audio sampling rate (default: 16000 Hz).
            word_callback (Optional[Callable[[str], None]]): Optional callback invoked when a word is decoded.
            min_phonemes_per_word (int): Minimum phonemes required to emit a word (default: 1).
            max_phonemes_per_word (int): Maximum phoneme length passed to Model 2 (default: 32).
            space_id (int): Phoneme token ID representing acoustic word boundaries (default: 8).
            blank_id (int): CTC blank token ID to ignore (default: 0).
            allowed_phoneme_range (Optional[Tuple[int, int]]): Valid phoneme token range [min, max] (default: (1, 111)).
            max_queue_size (int): Max queued words before blocking acoustic thread (default: 500).
        """
        # Model components
        self.model_a2p: Model_A2P = model_a2p
        self.model_pe: Model_PE = model_pe
        self.model_wd: Model_WD = model_wd

        # Audio stream settings
        self.sample_rate: int = sample_rate
        # Chunk size in samples: int, Role: acoustic window length, Rationale: 0.5s chunk at 16kHz = 8000 samples
        self.chunk_size_samples: int = int(chunk_duration_sec * sample_rate)

        # Phoneme segmentation parametrization
        self.min_phonemes_per_word: int = min_phonemes_per_word
        self.max_phonemes_per_word: int = max_phonemes_per_word
        self.space_id: int = space_id
        self.blank_id: int = blank_id
        self.allowed_phoneme_range: Optional[Tuple[int, int]] = allowed_phoneme_range

        # Thread-safe inter-thread communication buffers
        self.audio_buffer: AudioCircularBuffer = AudioCircularBuffer(capacity_samples=sample_rate * 60)
        self.word_queue: PhonemeWordQueue = PhonemeWordQueue(maxsize=max_queue_size)

        # Callback for real-time word streaming
        self.word_callback: Optional[Callable[[str], None]] = word_callback

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
    # Worker Thread 1: Model 1 Acoustic Processing
    # -------------------------------------------------------------------------
    def _acoustic_thread_loop(self) -> None:
        """
        Acoustic worker thread executing Model_A2P on audio stream chunks.
        Slices phonemes into words at space token (token 8) and forwards to queue.
        """
        self.model_a2p.eval()
        dev_a2p: torch.device = self.model_a2p.get_device()

        # Accumulated intra-word phoneme tokens waiting for space delimiter: List[int]
        current_word_phonemes: List[int] = []

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

            # Audio tensor: Tensor [1, T_samples], Role: batched audio for Model 1, Rationale: batch dimension
            audio_tensor: Tensor = audio_raw.unsqueeze(0).to(dev_a2p)

            t_a2p_start = time.perf_counter()
            with torch.no_grad():
                # Inference through Model 1
                a2p_out = self.model_a2p.forward_inference(audio_tensor)
                # Predicted collapsed phoneme token IDs: List[int]
                pred_tokens: List[int] = a2p_out["pred_tokens"][0] if a2p_out["pred_tokens"] else []
            self._total_acoustic_time += (time.perf_counter() - t_a2p_start)
            self._chunks_processed += 1

            # Segment phoneme tokens into words at space token
            for tok in pred_tokens:
                if tok == self.space_id:
                    # Space boundary encountered: push completed word to queue if valid length
                    if len(current_word_phonemes) >= self.min_phonemes_per_word:
                        clipped: List[int] = current_word_phonemes[: self.max_phonemes_per_word]
                        self.word_queue.push_word(list(clipped))
                    current_word_phonemes.clear()
                elif tok != self.blank_id:
                    # Verify phoneme is within allowed range
                    if self.allowed_phoneme_range is None or (
                        self.allowed_phoneme_range[0] <= tok <= self.allowed_phoneme_range[1]
                    ):
                        current_word_phonemes.append(tok)
                        # Avoid unbounded phoneme buffers
                        if len(current_word_phonemes) >= self.max_phonemes_per_word:
                            self.word_queue.push_word(list(current_word_phonemes))
                            current_word_phonemes.clear()

        # Flush any trailing phoneme word upon stream completion
        if len(current_word_phonemes) >= self.min_phonemes_per_word:
            clipped = current_word_phonemes[: self.max_phonemes_per_word]
            self.word_queue.push_word(list(clipped))
            current_word_phonemes.clear()

    # -------------------------------------------------------------------------
    # Worker Thread 2: Models 2 & 3 Linguistic Decoding
    # -------------------------------------------------------------------------
    def _decoder_thread_loop(self) -> None:
        """
        Decoder worker thread consuming phoneme words and executing Model_PE + Model_WD.
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
                decoded_words: List[str] = self.model_wd.forward_inference(z_word)

            self._total_decoder_time += (time.perf_counter() - t_dec_start)
            self._words_decoded += 1

            if decoded_words:
                emitted_word: str = decoded_words[0]
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
