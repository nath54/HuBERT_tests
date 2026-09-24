import pytest
import shutil
import tempfile
import torch
from pathlib import Path
from src.data.streaming_piper import PiperVoiceManager, VoiceQualityGuardian, PiperStreamingDataset, VoiceDepletionError


def test_voice_guardian_warning_and_blocking(tmp_path):
    blocklist_file = tmp_path / "test_blocked_voices.json"
    guardian = VoiceQualityGuardian(max_warnings=3, blocklist_file=blocklist_file)
    
    # 1. Initially no voices blocked
    assert not guardian.is_blocked("test_voice_1")
    assert guardian.warnings_count["test_voice_1"] == 0
    
    # 2. Add warning 1
    blocked = guardian.record_warning("test_voice_1", ["̃"], text_snippet="Bonjour un bon vin")
    assert not blocked
    assert guardian.warnings_count["test_voice_1"] == 1
    assert not guardian.is_blocked("test_voice_1")
    
    # 3. Add warning 2
    blocked = guardian.record_warning("test_voice_1", ["̃"], text_snippet="Les enfants chantent")
    assert not blocked
    assert guardian.warnings_count["test_voice_1"] == 2
    assert not guardian.is_blocked("test_voice_1")
    
    # 4. Add warning 3 -> Should block permanently!
    blocked = guardian.record_warning("test_voice_1", ["̃"], text_snippet="Chacun son tour")
    assert blocked
    assert guardian.warnings_count["test_voice_1"] == 3
    assert guardian.is_blocked("test_voice_1")
    assert "test_voice_1" in guardian.blocked_voices
    
    # 5. Verify saved to disk and reloaded
    assert blocklist_file.exists()
    reloaded_guardian = VoiceQualityGuardian(max_warnings=3, blocklist_file=blocklist_file)
    assert reloaded_guardian.is_blocked("test_voice_1")


def test_voice_manager_detects_and_replaces_defective_voice(tmp_path):
    blocklist_file = tmp_path / "test_manager_blocked.json"
    guardian = VoiceQualityGuardian(max_warnings=2, blocklist_file=blocklist_file)
    vm = PiperVoiceManager(guardian=guardian)
    
    initial_voice_count = len(vm.voice_models)
    assert initial_voice_count >= 30
    
    # Synthesize French text with nasal vowels
    # All currently active French voices (mls, siwis-medium, tom, upmc) must be 100% clean
    text = "Bonjour tout le monde, ceci est un test très important en français."
    waveform, voice_name, dur = vm.synthesize_to_tensor_16k(text, lang="fr")
    
    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 1
    assert len(waveform) > 0
    assert dur > 0.5
    assert not guardian.is_blocked(voice_name)
    assert guardian.warnings_count[voice_name] == 0


def test_voice_depletion_error_raised(tmp_path):
    """Verify that VoiceDepletionError is strictly raised when no clean voices remain for a language."""
    blocklist_file = tmp_path / "test_depletion_blocked.json"
    guardian = VoiceQualityGuardian(max_warnings=1, blocklist_file=blocklist_file)
    vm = PiperVoiceManager(guardian=guardian)
    
    # Artificially clear all French voices
    vm.fr_voices.clear()
    
    # Requesting a French voice MUST raise VoiceDepletionError
    with pytest.raises(VoiceDepletionError) as exc_info:
        vm.get_random_voice(lang="fr")
    
    err_msg = str(exc_info.value)
    assert "VOICE DEPLETION ERROR" in err_msg
    assert "FR" in err_msg
    assert "download_more_voices.py" in err_msg


def test_streaming_dataset_propagates_depletion_error(tmp_path):
    """Verify that PiperStreamingDataset raises VoiceDepletionError without swallowing it."""
    blocklist_file = tmp_path / "test_depletion_dataset.json"
    guardian = VoiceQualityGuardian(max_warnings=1, blocklist_file=blocklist_file)
    vm = PiperVoiceManager(guardian=guardian)
    vm.fr_voices.clear()
    
    class DummyFrenchSampler:
        def sample_sentence(self):
            return "Bonjour le monde", "fr"
            
    dataset = PiperStreamingDataset(voice_manager=vm, text_sampler=DummyFrenchSampler())
    
    iterator = iter(dataset)
    with pytest.raises(VoiceDepletionError):
        next(iterator)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
