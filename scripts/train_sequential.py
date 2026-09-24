"""Train HuBERT ASR on sequential phoneme acoustic representations."""

import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.utils.audio import synthesize_spoken_word


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    tokenizer = CharacterTokenizer()
    config = HuBERTConfig(
        vocab_size=tokenizer.vocab_size,
        encoder_layers=4,
        encoder_heads=4,
        encoder_embed_dim=256,
        encoder_ffn_dim=1024,
    )
    model = HuBERTForCTC(config).to(device)

    # Core vocabulary and phrase pool
    words = [
        "hello", "speech", "audio", "learn", "deep",
        "mind", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
        "bad", "bald", "good", "yes", "no",
        "cat", "dog", "run", "fast", "slow",
        "i am very bad", "i am very bald", "hello world",
        "deep speech", "audio learn"
    ]

    print(f"Generating synthetic training dataset across {len(words)} phonetic words & sentences...")
    dataset_samples = []
    for idx, w in enumerate(words):
        # Generate 4 variations per word with slightly different pitch / tempo
        for var in range(4):
            f0 = random.uniform(115.0, 165.0)
            wav = synthesize_spoken_word(w, f0=f0)
            dataset_samples.append({
                "id": f"{w}_{var}",
                "waveform": wav,
                "transcript": w,
                "sample_rate": 16000,
            })

    dataset = AudioASRDataset(dataset_samples, tokenizer, target_sample_rate=16000)
    collate_fn = AudioCollateFn(pad_token_id=tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=1e-4)
    epochs = 45

    print(f"Training HuBERT ASR for {epochs} epochs ({len(dataset)} samples)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            audio = batch["audio"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            outputs = model(
                audio=audio,
                audio_lengths=audio_lengths,
                targets=targets,
                target_lengths=target_lengths,
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_queries = ["hello", "speech", "bad", "bald", "one"]
                test_preds = []
                for q in test_queries:
                    q_wav = synthesize_spoken_word(q, f0=135.0).to(device)
                    out = model(q_wav.unsqueeze(0))
                    dec = model.decode_greedy(out["logits"])
                    test_preds.append(f"'{q}'->'{tokenizer.decode(dec[0])}'")

                print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.3f} | " + " | ".join(test_preds))

    # Save to checkpoints/best_model.pt
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, "checkpoints/best_model.pt")
    print("\n[Done] Successfully saved converged checkpoint to checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
