# Standardized Cross-Architecture Benchmark Report

**Date**: 2026-09-30 20:28:21  
**Model Tiers**: `medium` | **Training Steps**: `4,000` | **Evaluation Interval**: every `200` steps  
**Training Split**: `benchmark_train.json` (~95% LibriSpeech Clean-100)  
**Validation Split**: `benchmark_val.json` (~5% Held-Out LibriSpeech Clean-100)  
**Test Benchmark**: `librispeech_test_clean.json` (Untainted Test-Clean)  
**Summary**: 8 Successful / 0 Failed / 8 Total Runs  
**Total Suite Runtime**: 479.5 minutes  

## Executive Summary & Model Selection Protocol
1. **Fair & Scientific Comparison**: All architectures trained on identical audio frames and random seeds with identical learning rates and batch sizes.
2. **Strict Cheat-Free Policy**: The test-clean split was never accessed during training. Models were evaluated every 200 steps solely on the held-out validation set.
3. **Zero Waste / SSD Protection**: Only the single best model weights (`best_checkpoint.pt`) based on validation PER were preserved on disk.
4. **Fault Tolerance**: Any model encountering out-of-memory or errors is safely caught, documented with its exact error, and the suite automatically proceeds to the next model.
5. **Final Scoring**: The best validation model was restored at the conclusion of training to obtain the definitive test-clean score.

## Consolidated Benchmark Results

| Architecture | Tier | Best Val PER | Best Step | Test PER (Greedy) | Test Lexicon PER | Test CER | Audio Hours | Status / Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **phono_v6_4_gated_diffusion** | medium | 18.27% | 2800 | **14.63%** | 16.78% | 14.63% | 56.2h | `checkpoints/phono_v6_4_gated_diffusion/medium/best_checkpoint.pt` |
| **phono_v6_3_diffusion** | medium | 20.58% | 3600 | **16.05%** | 19.46% | 16.05% | 56.3h | `checkpoints/phono_v6_3_diffusion/medium/best_checkpoint.pt` |
| **phono_v4_scaled** | medium | 32.73% | 3400 | **27.49%** | 36.96% | 27.49% | 56.4h | `checkpoints/phono_v4_scaled/medium/best_checkpoint.pt` |
| **phono_v6_1_moe** | medium | 35.49% | 3800 | **28.60%** | 33.83% | 28.60% | 56.4h | `checkpoints/phono_v6_1_moe/medium/best_checkpoint.pt` |
| **phono_v5_beam** | medium | 35.50% | 3600 | **29.77%** | 32.64% | 29.77% | 56.6h | `checkpoints/phono_v5_beam/medium/best_checkpoint.pt` |
| **phono_v2_specaugment** | medium | 37.08% | 4000 | **30.38%** | 34.75% | 30.38% | 56.3h | `checkpoints/phono_v2_specaugment/medium/best_checkpoint.pt` |
| **phono_v3_hybrid** | medium | 39.34% | 4000 | **34.83%** | 40.89% | 34.83% | 56.4h | `checkpoints/phono_v3_hybrid/medium/best_checkpoint.pt` |
| **phono_v1_frontend** | medium | 43.51% | 3800 | **37.98%** | 30.37% | 37.98% | 56.5h | `checkpoints/phono_v1_frontend/medium/best_checkpoint.pt` |

```
Test PER Ranking (Lower is Better):
  phono_v6_4_gated_diffusion [medium]: █████ 14.63%
  phono_v6_3_diffusion [medium]   : ██████ 16.05%
  phono_v4_scaled [medium]        : ██████████ 27.49%
  phono_v6_1_moe [medium]         : ███████████ 28.60%
  phono_v5_beam [medium]          : ███████████ 29.77%
  phono_v2_specaugment [medium]   : ████████████ 30.38%
  phono_v3_hybrid [medium]        : █████████████ 34.83%
  phono_v1_frontend [medium]      : ███████████████ 37.98%
```
