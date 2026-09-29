# Standardized Cross-Architecture Benchmark Report

**Date**: 2026-09-29 19:11:26  
**Model Tiers**: `medium` | **Training Steps**: `4,000` | **Evaluation Interval**: every `200` steps  
**Training Split**: `benchmark_train.json` (~95% LibriSpeech Clean-100)  
**Validation Split**: `benchmark_val.json` (~5% Held-Out LibriSpeech Clean-100)  
**Test Benchmark**: `librispeech_test_clean.json` (Untainted Test-Clean)  
**Summary**: 7 Successful / 0 Failed / 7 Total Runs  
**Total Suite Runtime**: 524.5 minutes  

## Executive Summary & Model Selection Protocol
1. **Fair & Scientific Comparison**: All architectures trained on identical audio frames and random seeds with identical learning rates and batch sizes.
2. **Strict Cheat-Free Policy**: The test-clean split was never accessed during training. Models were evaluated every 200 steps solely on the held-out validation set.
3. **Zero Waste / SSD Protection**: Only the single best model weights (`best_checkpoint.pt`) based on validation PER were preserved on disk.
4. **Fault Tolerance**: Any model encountering out-of-memory or errors is safely caught, documented with its exact error, and the suite automatically proceeds to the next model.
5. **Final Scoring**: The best validation model was restored at the conclusion of training to obtain the definitive test-clean score.

## Consolidated Benchmark Results

| Architecture | Tier | Best Val PER | Best Step | Test PER (Greedy) | Test Lexicon PER | Test CER | Audio Hours | Status / Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **phono_v6_3_diffusion** | medium | 20.58% | 3600 | **16.05%** | 19.46% | 16.05% | 56.3h | `checkpoints/phono_v6_3_diffusion/medium/best_checkpoint.pt` |
| **phono_v6_2_sparse** | medium | 24.82% | 4000 | **19.13%** | 20.27% | 19.13% | 56.4h | `checkpoints/phono_v6_2_sparse/medium/best_checkpoint.pt` |
| **phono_v6_1_moe** | medium | 47.06% | 4000 | **39.37%** | 44.46% | 39.37% | 56.4h | `checkpoints/phono_v6_1_moe/medium/best_checkpoint.pt` |
| **phono_hubert_dual** | medium | 97.37% | 200 | **94.97%** | 81.24% | 94.97% | 56.7h | `checkpoints/phono_hubert_dual/medium/best_checkpoint.pt` |
| **phono_hubert_recursive** | medium | 95.99% | 200 | **95.74%** | 142.69% | 95.74% | 56.5h | `checkpoints/phono_hubert_recursive/medium/best_checkpoint.pt` |
| **phono_hubert** | medium | 99.04% | 400 | **99.34%** | 97.34% | 99.34% | 56.3h | `checkpoints/phono_hubert/medium/best_checkpoint.pt` |
| **phono_hubert_hierarchical** | medium | 100.00% | 200 | **100.00%** | 100.00% | 100.00% | 56.4h | `checkpoints/phono_hubert_hierarchical/medium/best_checkpoint.pt` |

```
Test PER Ranking (Lower is Better):
  phono_v6_3_diffusion [medium]   : ██████ 16.05%
  phono_v6_2_sparse [medium]      : ███████ 19.13%
  phono_v6_1_moe [medium]         : ███████████████ 39.37%
  phono_hubert_dual [medium]      : █████████████████████████████████████ 94.97%
  phono_hubert_recursive [medium] : ██████████████████████████████████████ 95.74%
  phono_hubert [medium]           : ███████████████████████████████████████ 99.34%
  phono_hubert_hierarchical [medium]: ████████████████████████████████████████ 100.00%
```
