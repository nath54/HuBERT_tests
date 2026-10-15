"""
Thread-Safe Streaming Buffers for Dual-Threaded Speech Recognition.
Provides:
1. AudioCircularBuffer: Thread-safe ring buffer for incoming 16kHz raw audio samples.
2. PhonemeWordQueue: Thread-safe queue for phoneme-sliced words passed from Model 1 to Model 2+3.
"""

# Import Modules
from typing import List, Optional, Tuple, Union
import queue
import threading
import torch
from torch import Tensor


class AudioCircularBuffer:
    """
    Thread-safe circular ring buffer for streaming 16kHz audio waveforms.
    Allows concurrent writing of audio chunks by the producer (audio ingest / microphone)
    and reading of chunked frames by the consumer (Model 1 acoustic encoder thread).
    """

    def __init__(
        self,
        capacity_samples: int = 16000 * 30,  # 30 seconds at 16kHz
    ) -> None:
        """
        Initialize the AudioCircularBuffer.

        Args:
            capacity_samples (int): Maximum number of audio samples stored in ring buffer.
        """
        # Buffer capacity: int, Role: maximum buffer length in samples, Rationale: prevent memory blowup
        self.capacity_samples: int = capacity_samples

        # Internal audio buffer: Tensor [capacity_samples], Role: circular storage array, Rationale: zero-allocation ring
        self.buffer: Tensor = torch.zeros(capacity_samples, dtype=torch.float32)

        # Write position pointer: int, Role: current insertion index, Rationale: circular wrap-around tracking
        self.write_pos: int = 0

        # Read position pointer: int, Role: current consumption index, Rationale: circular read tracking
        self.read_pos: int = 0

        # Available unread sample count: int, Role: samples ready for acoustic model, Rationale: synchronization
        self.available_samples: int = 0

        # Concurrency primitives
        self._lock: threading.Lock = threading.Lock()
        self._not_empty: threading.Condition = threading.Condition(self._lock)

    def write(self, audio_chunk: Tensor) -> int:
        """
        Writes incoming audio samples into the ring buffer.

        Args:
            audio_chunk (Tensor): 1D tensor [T] or [1, T] containing float32 audio samples.

        Returns:
            int: Number of samples successfully written.
        """
        # Ensure 1D flat tensor: Tensor [T], Role: linear sample array, Rationale: ring indexing
        samples: Tensor = audio_chunk.detach().cpu().flatten().float()
        num_samples: int = samples.shape[0]

        if num_samples == 0:
            return 0

        with self._not_empty:
            if num_samples > self.capacity_samples:
                # Truncate to newest samples if chunk exceeds entire buffer capacity
                samples = samples[-self.capacity_samples :]
                num_samples = self.capacity_samples

            # Linear insertion or wrap-around write
            end_pos: int = self.write_pos + num_samples
            if end_pos <= self.capacity_samples:
                self.buffer[self.write_pos : end_pos] = samples
            else:
                first_part_len: int = self.capacity_samples - self.write_pos
                self.buffer[self.write_pos : self.capacity_samples] = samples[:first_part_len]
                self.buffer[0 : end_pos % self.capacity_samples] = samples[first_part_len:]

            self.write_pos = (self.write_pos + num_samples) % self.capacity_samples
            self.available_samples = min(self.capacity_samples, self.available_samples + num_samples)

            # Notify waiting consumer thread that new audio has arrived
            self._not_empty.notify_all()

        return num_samples

    def read(
        self,
        num_samples: int,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[Tensor]:
        """
        Reads a contiguous slice of audio samples from the buffer.

        Args:
            num_samples (int): Desired number of samples to read.
            block (bool): Whether to block until requested samples become available.
            timeout (Optional[float]): Maximum wait time in seconds if blocking.

        Returns:
            Optional[Tensor]: 1D tensor [num_samples] if available, otherwise None.
        """
        with self._not_empty:
            if block and self.available_samples < num_samples:
                # Wait until enough samples are present or timeout occurs
                self._not_empty.wait_for(lambda: self.available_samples >= num_samples, timeout=timeout)

            if self.available_samples < num_samples:
                return None

            # Extracted audio: Tensor [num_samples], Role: frame for acoustic model, Rationale: chunked inference
            out_tensor: Tensor = torch.empty(num_samples, dtype=torch.float32)

            end_pos: int = self.read_pos + num_samples
            if end_pos <= self.capacity_samples:
                out_tensor.copy_(self.buffer[self.read_pos : end_pos])
            else:
                first_part_len: int = self.capacity_samples - self.read_pos
                out_tensor[:first_part_len].copy_(self.buffer[self.read_pos : self.capacity_samples])
                out_tensor[first_part_len:].copy_(self.buffer[0 : end_pos % self.capacity_samples])

            self.read_pos = (self.read_pos + num_samples) % self.capacity_samples
            self.available_samples -= num_samples

            return out_tensor

    def read_all_available(self) -> Optional[Tensor]:
        """
        Reads all currently available unread audio samples without blocking.

        Returns:
            Optional[Tensor]: 1D tensor [available_samples] or None if buffer is empty.
        """
        with self._lock:
            if self.available_samples == 0:
                return None
            count: int = self.available_samples
            return self.read(num_samples=count, block=False)

    def get_available_samples(self) -> int:
        """
        Returns the number of unread audio samples in the buffer.

        Returns:
            int: Sample count.
        """
        with self._lock:
            return self.available_samples

    def clear(self) -> None:
        """
        Resets read/write pointers and empties the buffer.
        """
        with self._lock:
            self.write_pos = 0
            self.read_pos = 0
            self.available_samples = 0


class PhonemeWordQueue:
    """
    Thread-safe FIFO queue for passing word-segmented phoneme sequences from Model 1 (A2P)
    to Models 2 & 3 (PE + WD) in the dual-threaded pipeline.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        """
        Initialize the PhonemeWordQueue.

        Args:
            maxsize (int): Maximum number of unconsumed words in queue.
        """
        # Internal thread-safe FIFO queue
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

        # Sentinel object indicating end-of-stream
        self._sentinel: object = object()

    def push_word(self, phoneme_ids: List[int]) -> None:
        """
        Pushes a sliced phoneme sequence representing a completed word.

        Args:
            phoneme_ids (List[int]): Intra-word phoneme token IDs without boundary spaces.
        """
        if phoneme_ids:
            self._queue.put(phoneme_ids)

    def pop_word(self, block: bool = True, timeout: Optional[float] = None) -> Optional[List[int]]:
        """
        Pops the next available phoneme word sequence.

        Args:
            block (bool): Whether to block waiting for items.
            timeout (Optional[float]): Timeout in seconds.

        Returns:
            Optional[List[int]]: Phoneme ID list, or None if queue is empty or end-of-stream reached.
        """
        try:
            item = self._queue.get(block=block, timeout=timeout)
            if item is self._sentinel:
                # Re-put sentinel so future calls also receive EOS
                self._queue.put(self._sentinel)
                return None
            return item
        except queue.Empty:
            return None

    def push_end_of_stream(self) -> None:
        """
        Signals end of audio stream to the consumer thread.
        """
        self._queue.put(self._sentinel)

    def is_empty(self) -> bool:
        """
        Returns True if queue is currently empty.

        Returns:
            bool: Emptiness flag.
        """
        return self._queue.empty()

    def size(self) -> int:
        """
        Returns the number of words waiting in the queue.

        Returns:
            int: Number of queued words.
        """
        return self._queue.qsize()
