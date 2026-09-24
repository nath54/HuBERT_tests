"""Explainable AI (XAI) Suite for HuBERT ASR."""

from src.xai.captum_gradients import (
    AudioGradientExplainer,
    TokenLogitModelWrapper,
    CTCSequenceLossWrapper,
)
from src.xai.naps import (
    NeuronActivationProfiler,
    NeuronSaliencyAnalyzer,
    ActivationPatcher,
)
from src.xai.probes import (
    LinearProbe,
    AcousticInversionDecoder,
    LayerwiseProbeTrainer,
)
from src.xai.visualizer import (
    plot_waveform_attribution,
    plot_spectrogram_and_attribution,
    plot_neuron_selectivity_heatmap,
    plot_layer_probing_curve,
    plot_attention_map,
)

__all__ = [
    "AudioGradientExplainer",
    "TokenLogitModelWrapper",
    "CTCSequenceLossWrapper",
    "NeuronActivationProfiler",
    "NeuronSaliencyAnalyzer",
    "ActivationPatcher",
    "LinearProbe",
    "AcousticInversionDecoder",
    "LayerwiseProbeTrainer",
    "plot_waveform_attribution",
    "plot_spectrogram_and_attribution",
    "plot_neuron_selectivity_heatmap",
    "plot_layer_probing_curve",
    "plot_attention_map",
]
