"""
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

    models = {'a2p': '/home/nathan/github/audiolearn/src/test_export_pixel10/model1_a2p_pixel10.onnx', 'pe': '/home/nathan/github/audiolearn/src/test_export_pixel10/model2_pe_pixel10.onnx', 'wd': '/home/nathan/github/audiolearn/src/test_export_pixel10/model3_wd_pixel10.onnx'}
    for name, onnx_file in models.items():
        if not os.path.isfile(onnx_file):
            print(f"Skipping {name}: {onnx_file} not found")
            continue
        print(f"Converting {name} from {onnx_file} to TFLite...")
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
        print(f"Saved optimized TFLite model: {tflite_path}")

if __name__ == "__main__":
    convert()
