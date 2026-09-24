import pytest
import shutil
import tempfile
import torch
from pathlib import Path
from src.data.streaming_piper import PiperVoiceManager, VoiceQualityGuardian, PiperStreamingDataset

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

if __name__ == "__main__":
    pytest.main(["-v", __file__])
