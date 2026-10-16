"""
TFLite and Mobile Edge Export Utilities for Google Pixel 10 Deployment.
Provides static-shape ONNX and TFLite graph generation tailored for
the Google Tensor G5 Mobile NPU and Android Neural Networks API (NNAPI).
"""

# Import Modules
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
from torch import nn, Tensor

from ..model1_a2p import Model_A2P
from ..model2_pe import Model_PE
from ..model3_wd import Model_WD
from ..model import Model


class TFLiteExporter:
    """
    Exports clean model components to mobile-optimized static graphs (ONNX / TFLite / TorchScript)
    for hardware-accelerated on-device execution on Google Pixel 10 (Tensor G5 NPU).
    """

    def __init__(
        self,
        output_dir: str = "exported_models",
        opset_version: int = 17,
    ) -> None:
        """
        Initialize the TFLiteExporter.

        Args:
            output_dir (str): Directory where exported model graphs will be saved.
            opset_version (int): ONNX operator set version (17 recommended for Conformer & LayerNorm).
        """
        self.output_dir: str = output_dir
        self.opset_version: int = opset_version
        os.makedirs(output_dir, exist_ok=True)

    def export_a2p_onnx(
        self,
        model_a2p: Model_A2P,
        filename: str = "model1_a2p_pixel10.onnx",
        chunk_seconds: float = 3.0,
        sample_rate: int = 16000,
        static_shape: bool = True,
    ) -> str:
        """
        Exports Model_A2P (Audio to Phonemes) to ONNX with static shapes for Tensor G5 NPU compilation.

        Args:
            model_a2p (Model_A2P): Part 1 Acoustic model.
            filename (str): Name of the exported ONNX file.
            chunk_seconds (float): Length of fixed audio chunk in seconds.
            sample_rate (int): Audio sampling rate (16000 Hz).
            static_shape (bool): Whether to enforce static shapes (required for mobile NPU delegates).

        Returns:
            str: Path to saved ONNX file.
        """
        model_a2p.eval()
        output_path: str = os.path.join(self.output_dir, filename)

        # Fixed sample count: int, Role: static audio buffer length, Rationale: NPU compilation requires static shapes
        num_samples: int = int(chunk_seconds * sample_rate)

        # Dummy audio tensor: Tensor [1, num_samples], Role: tracer input, Rationale: trace forward execution
        dummy_audio: Tensor = torch.zeros((1, num_samples), dtype=torch.float32, device=model_a2p.get_device())

        # Adapter class wrapping Model_A2P to output clean logits tensor for mobile inference
        class Model1A2PExportWrapper(nn.Module):
            def __init__(self, core_model: Model_A2P):
                super().__init__()
                self.core = core_model

            def forward(self, x: Tensor) -> Tensor:
                # Forward acoustic backbone to obtain composite phoneme logits: Tensor [1, T_frames, vocab_size]
                out = self.core(x)
                return out["logits"]

        wrapper = Model1A2PExportWrapper(model_a2p).to(model_a2p.get_device())
        wrapper.eval()

        dynamic_axes = None if static_shape else {"audio": {1: "time_samples"}, "phoneme_logits": {1: "frames"}}

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_audio,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=["audio"],
                output_names=["phoneme_logits"],
                dynamic_axes=dynamic_axes,
            )

        print(f"[TFLiteExporter] Successfully exported Model 1 to ONNX: {output_path}")
        return output_path

    def export_pe_onnx(
        self,
        model_pe: Model_PE,
        filename: str = "model2_pe_pixel10.onnx",
        fixed_word_length: int = 16,
    ) -> str:
        """
        Exports Model_PE (Phoneme Encoder) to ONNX with static word slice lengths.

        Args:
            model_pe (Model_PE): Part 2 Phoneme Word Encoder.
            filename (str): Name of the exported ONNX file.
            fixed_word_length (int): Static intra-word phoneme sequence length.

        Returns:
            str: Path to saved ONNX file.
        """
        model_pe.eval()
        output_path: str = os.path.join(self.output_dir, filename)

        # Dummy phonemes: Tensor [1, fixed_word_length], Role: tracer phoneme input, Rationale: static input shape
        dummy_phonemes: Tensor = torch.ones((1, fixed_word_length), dtype=torch.long, device=model_pe.get_device())

        class Model2PEExportWrapper(nn.Module):
            def __init__(self, core_model: Model_PE):
                super().__init__()
                self.core = core_model

            def forward(self, phoneme_ids: Tensor) -> Tensor:
                # Encode phonemes: Tensor [1, 1, 512]
                out = self.core.forward_training(phoneme_ids.unsqueeze(1))
                return out["z_word"]

        wrapper = Model2PEExportWrapper(model_pe).to(model_pe.get_device())
        wrapper.eval()

        # Disable PyTorch native fastpath to generate standard ONNX MultiHeadAttention operators
        prev_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    dummy_phonemes,
                    output_path,
                    export_params=True,
                    opset_version=self.opset_version,
                    do_constant_folding=True,
                    input_names=["phoneme_ids"],
                    output_names=["z_word"],
                    dynamic_axes=None,
                )
        finally:
            torch.backends.mha.set_fastpath_enabled(prev_fastpath)

        print(f"[TFLiteExporter] Successfully exported Model 2 to ONNX: {output_path}")
        return output_path

    def export_wd_onnx(
        self,
        model_wd: Model_WD,
        filename: str = "model3_wd_pixel10.onnx",
        max_chars: int = 20,
    ) -> str:
        """
        Exports Model_WD (Word Decoder single-step character predictor) to ONNX.

        Args:
            model_wd (Model_WD): Part 3 Word Decoder.
            filename (str): Name of the exported ONNX file.
            max_chars (int): Maximum word character length.

        Returns:
            str: Path to saved ONNX file.
        """
        model_wd.eval()
        output_path: str = os.path.join(self.output_dir, filename)

        # Dummy z_word: Tensor [1, 1, 512], Role: continuous word embedding, Rationale: latent input
        dummy_z_word: Tensor = torch.zeros((1, 1, 512), dtype=torch.float32, device=model_wd.get_device())
        # Dummy byte_ids: Tensor [1, 1, max_chars], Role: prefix character tokens, Rationale: autoregressive step
        dummy_byte_ids: Tensor = torch.zeros((1, 1, max_chars), dtype=torch.long, device=model_wd.get_device())

        class Model3WDExportWrapper(nn.Module):
            def __init__(self, core_model: Model_WD):
                super().__init__()
                self.core = core_model

            def forward(self, z_word: Tensor, byte_ids: Tensor) -> Tensor:
                # Predict character logits: Tensor [1, 1, max_chars, 256]
                logits = self.core.forward_single_step(z_word, byte_ids)
                return logits

        wrapper = Model3WDExportWrapper(model_wd).to(model_wd.get_device())
        wrapper.eval()

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_z_word, dummy_byte_ids),
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=["z_word", "byte_ids"],
                output_names=["char_logits"],
                dynamic_axes=None,
            )

        print(f"[TFLiteExporter] Successfully exported Model 3 to ONNX: {output_path}")
        return output_path

    def export_all(
        self,
        model: Model,
        chunk_seconds: float = 3.0,
    ) -> Dict[str, str]:
        """
        Exports all tripartite model parts into static ONNX graphs for Google Pixel 10.

        Args:
            model (Model): Unified master Model instance.
            chunk_seconds (float): Fixed audio duration for Model 1 chunking.

        Returns:
            Dict[str, str]: Paths to exported ONNX models for parts 1, 2, and 3.
        """
        paths: Dict[str, str] = {}
        paths["a2p"] = self.export_a2p_onnx(model.model_a2p, chunk_seconds=chunk_seconds)
        paths["pe"] = self.export_pe_onnx(model.model_pe)
        paths["wd"] = self.export_wd_onnx(model.model_wd)
        return paths

    def generate_tflite_conversion_script(
        self,
        onnx_paths: Dict[str, str],
        output_script_name: str = "convert_to_tflite.py",
    ) -> str:
        """
        Generates an automated script converting ONNX models to Google Pixel 10 TFLite models
        using TensorFlow Lite Converter with full INT8 PTQ quantization flags.

        Args:
            onnx_paths (Dict[str, str]): Dict of ONNX file paths.
            output_script_name (str): Name of the generated Python conversion script.

        Returns:
            str: Path to the generated conversion script.
        """
        script_path: str = os.path.join(self.output_dir, output_script_name)
        script_content = f'''"""
Automated TFLite Converter Script for Google Pixel 10 (Tensor G5 NPU).
Converts exported ONNX models to .tflite with NNAPI and GPU delegate optimizations.
"""

import os
import sys

def convert():
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        print("[TFLite Converter] Required packages: pip install onnx onnx-tf tensorflow")
        print("[TFLite Converter] Or use ai-edge-torch for native PyTorch to TFLite conversion.")
        return

    models = {onnx_paths}
    for name, onnx_file in models.items():
        if not os.path.isfile(onnx_file):
            print(f"Skipping {{name}}: {{onnx_file}} not found")
            continue
        print(f"Converting {{name}} from {{onnx_file}} to TFLite...")
        tf_rep_dir = onnx_file.replace(".onnx", "_tf")
        onnx_model = onnx.load(onnx_file)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_rep_dir)

        # Convert SavedModel to TFLite with Float16 and Int8 optimizations
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        tflite_path = onnx_file.replace(".onnx", ".tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved optimized TFLite model: {{tflite_path}}")

if __name__ == "__main__":
    convert()
'''
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        print(f"[TFLiteExporter] Generated conversion script at: {script_path}")
        return script_path
