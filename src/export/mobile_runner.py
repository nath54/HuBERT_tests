"""
Mobile Edge Inference Runner for Google Pixel 10 (Tensor G5 NPU).
Executes exported static-graph models using ONNX Runtime (with NNAPI / CPU delegates)
or TensorFlow Lite runtime, simulating on-device smartphone latency and throughput.
"""

# Import Modules
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import time
import numpy as np
import torch
from torch import Tensor


class Pixel10MobileRunner:
    """
    Simulates and executes the tripartite ASR pipeline on mobile edge hardware
    (Google Pixel 10 Tensor G5 NPU / CPU) using ONNX Runtime or TFLite.
    """

    def __init__(
        self,
        a2p_model_path: str,
        pe_model_path: str,
        wd_model_path: str,
        execution_provider: str = "CPUExecutionProvider",  # or 'NNAPIExecutionProvider' on Android
    ) -> None:
        """
        Initialize the Pixel10MobileRunner.

        Args:
            a2p_model_path (str): File path to Model 1 ONNX/TFLite model.
            pe_model_path (str): File path to Model 2 ONNX/TFLite model.
            wd_model_path (str): File path to Model 3 ONNX/TFLite model.
            execution_provider (str): Hardware execution delegate (e.g. NNAPIExecutionProvider).
        """
        self.a2p_model_path: str = a2p_model_path
        self.pe_model_path: str = pe_model_path
        self.wd_model_path: str = wd_model_path
        self.execution_provider: str = execution_provider

        # Initialize ONNX runtime sessions
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        providers = [execution_provider]
        if execution_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")

        print(f"[Pixel10MobileRunner] Loading Model 1 from {a2p_model_path}...")
        self.sess_a2p = ort.InferenceSession(a2p_model_path, sess_options, providers=providers)

        print(f"[Pixel10MobileRunner] Loading Model 2 from {pe_model_path}...")
        self.sess_pe = ort.InferenceSession(pe_model_path, sess_options, providers=providers)

        print(f"[Pixel10MobileRunner] Loading Model 3 from {wd_model_path}...")
        self.sess_wd = ort.InferenceSession(wd_model_path, sess_options, providers=providers)

        # Inspect input shapes
        self.a2p_input_name: str = self.sess_a2p.get_inputs()[0].name
        self.a2p_expected_samples: int = self.sess_a2p.get_inputs()[0].shape[1]
        self.pe_input_name: str = self.sess_pe.get_inputs()[0].name
        self.pe_expected_len: int = self.sess_pe.get_inputs()[0].shape[1]

    def transcribe_audio_chunk(
        self,
        audio_chunk: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Transcribes a static audio chunk through the mobile pipeline.

        Args:
            audio_chunk (np.ndarray): 1D float32 audio waveform.

        Returns:
            Dict[str, Any]:
                - 'transcript': Final decoded string
                - 'phonemes': Extracted phoneme sequence
                - 'latency_ms': Breakdown of execution times in milliseconds
                - 'rtf': Real-Time Factor (inference_time / audio_duration)
        """
        t0 = time.perf_counter()

        # 1. Pad or truncate audio to expected static input size for Tensor G5 NPU
        if audio_chunk.ndim == 1:
            audio_np = audio_chunk.reshape(1, -1).astype(np.float32)
        else:
            audio_np = audio_chunk.astype(np.float32)

        cur_len = audio_np.shape[1]
        target_len = self.a2p_expected_samples
        if cur_len < target_len:
            pad = np.zeros((1, target_len - cur_len), dtype=np.float32)
            audio_input = np.concatenate([audio_np, pad], axis=1)
        else:
            audio_input = audio_np[:, :target_len]

        # 2. Execute Model 1 (Acoustic -> Phoneme Logits)
        t_a2p_start = time.perf_counter()
        outputs_a2p = self.sess_a2p.run(None, {self.a2p_input_name: audio_input})
        logits_a2p = outputs_a2p[0]  # Shape: [1, T_frames, 112]
        t_a2p_end = time.perf_counter()

        # Greedy CTC collapse: argmax along vocab axis
        pred_ids = np.argmax(logits_a2p[0], axis=-1)  # Shape: [T_frames]
        collapsed_tokens: List[int] = []
        prev_tok: int = -1
        for tok in pred_ids:
            if tok != prev_tok:
                if tok != 0:  # Skip blank
                    collapsed_tokens.append(int(tok))
                prev_tok = tok

        # 3. Slice phonemes into words by space token (8)
        word_slices: List[List[int]] = []
        curr_word: List[int] = []
        for tok in collapsed_tokens:
            if tok == 8:
                if curr_word:
                    word_slices.append(curr_word)
                    curr_word = []
            else:
                curr_word.append(tok)
        if curr_word:
            word_slices.append(curr_word)

        # 4. Execute Models 2 & 3 for each word slice
        t_dec_start = time.perf_counter()
        decoded_words: List[str] = []

        for w_phonemes in word_slices:
            if not w_phonemes:
                continue

            # Model 2: Prepare static phoneme input
            pe_in = np.ones((1, self.pe_expected_len), dtype=np.int64)
            copy_len = min(len(w_phonemes), self.pe_expected_len)
            pe_in[0, :copy_len] = w_phonemes[:copy_len]

            # Run Model 2 -> z_word [1, 1, 512]
            z_word_out = self.sess_pe.run(None, {self.pe_input_name: pe_in})[0]

            # Model 3: Greedy rollout using static forward_single_step
            byte_seq = [2]  # <BOS>
            max_chars = 20
            for _ in range(max_chars):
                byte_in = np.zeros((1, 1, max_chars), dtype=np.int64)
                byte_in[0, 0, : len(byte_seq)] = byte_seq[:max_chars]

                char_logits = self.sess_wd.run(
                    None,
                    {"z_word": z_word_out.astype(np.float32), "byte_ids": byte_in},
                )[0]

                # Position to read: index len(byte_seq)-1
                step_idx = min(len(byte_seq) - 1, max_chars - 1)
                next_char_id = int(np.argmax(char_logits[0, 0, step_idx, :]))

                if next_char_id in (0, 3):  # PAD or EOS
                    break
                byte_seq.append(next_char_id)

            # Convert byte IDs (offset by 4) to UTF-8 characters
            char_bytes = bytes([b - 4 for b in byte_seq[1:] if b >= 4])
            try:
                decoded_word_str = char_bytes.decode("utf-8", errors="ignore")
            except Exception:
                decoded_word_str = ""
            if decoded_word_str:
                decoded_words.append(decoded_word_str)

        t_dec_end = time.perf_counter()
        t_total = time.perf_counter() - t0

        audio_sec = cur_len / 16000.0
        rtf = t_total / audio_sec if audio_sec > 0 else 0.0

        return {
            "transcript": " ".join(decoded_words).strip(),
            "phonemes": collapsed_tokens,
            "latency_ms": {
                "model1_a2p_ms": round((t_a2p_end - t_a2p_start) * 1000.0, 2),
                "model2_3_dec_ms": round((t_dec_end - t_dec_start) * 1000.0, 2),
                "total_ms": round(t_total * 1000.0, 2),
            },
            "rtf": round(rtf, 4),
        }
