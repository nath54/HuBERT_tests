"""
Weights and Checkpoint Manager for Tripartite Speech Recognition Models.
Features:
- Version and model size tier management (tiny, small, base, large)
- Automated weight downloading from GitHub Releases or remote URLs when local checkpoints are missing
- Checksum verification (SHA-256) and local cache management
- Easy configuration of custom release asset URLs.
"""

# Import Modules
from dataclasses import dataclass, field
import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request

# Default local checkpoint storage directory
DEFAULT_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))


@dataclass
class WeightMetadata:
    """
    Specification for a model checkpoint release asset.
    """
    # Key identifier e.g. 'a2p_base_v8.0'
    key: str

    # Local filename e.g. 'phono_v8_acoustic_multilingual_base.pt'
    filename: str

    # Target model part: 'a2p' (Model 1), 'p2g' (Models 2+3), or 'cnn' (Acoustic frontend)
    part: str

    # Model size tier: 'tiny', 'small', 'base', 'large'
    size: str

    # Model version: 'v8.0', 'v8.1', 'latest'
    version: str

    # Download URL (GitHub Release asset link)
    url: Optional[str] = None

    # Expected SHA-256 checksum for integrity verification
    sha256: Optional[str] = None

    # File size in megabytes
    size_mb: float = 0.0

    # Human-readable description
    description: str = ""

    # Known fallback local search paths
    fallback_local_paths: List[str] = field(default_factory=list)


class WeightsManager:
    """
    Manages discovery, verification, and automatic downloading of pretrained weights
    across model parts (A2P, P2G, CNN), model size tiers (tiny, small, base, large), and versions.
    """

    # Centralized Checkpoint Registry
    REGISTRY: Dict[str, WeightMetadata] = {
        # --- CNN Pretrained Filter Weights ---
        "cnn_frontend": WeightMetadata(
            key="cnn_frontend",
            filename="pretrained_cnn_ls960.pt",
            part="cnn",
            size="base",
            version="v1.0",
            url=os.environ.get("PHONO_URL_CNN", None),
            sha256="f76443cc0a54c62711ca34792cbf6cf3e97741910f3d6d571ad90b9d7ef5bf16",
            size_mb=17.0,
            description="7-layer 1D CNN feature extractor downsampling 16kHz audio 320x to 50Hz",
            fallback_local_paths=[
                "data/pretrained_cnn_ls960.pt",
                "checkpoints/pretrained_cnn_ls960.pt",
            ],
        ),

        # --- Model 1: Acoustic Conformer MoE (A2P) ---
        "a2p_base_v8.0": WeightMetadata(
            key="a2p_base_v8.0",
            filename="phono_v8_acoustic_multilingual_base.pt",
            part="a2p",
            size="base",
            version="v8.0",
            url=os.environ.get("PHONO_URL_A2P_BASE", None),
            sha256=None,
            size_mb=778.0,
            description="12-layer Conformer Backbone with DeepSeek MoE, Phoneme Engram Memory, and CTC heads",
            fallback_local_paths=[
                "checkpoints/phono_v8_acoustic_multilingual_sota/latest_v8_acoustic.pt",
                "checkpoints/phono_v8_acoustic_multilingual_sota/best_v8_acoustic.pt",
                "checkpoints/phono_v8_acoustic_multilingual_base.pt",
            ],
        ),
        "a2p_tiny_v8.0": WeightMetadata(
            key="a2p_tiny_v8.0",
            filename="phono_v8_acoustic_tiny.pt",
            part="a2p",
            size="tiny",
            version="v8.0",
            url=os.environ.get("PHONO_URL_A2P_TINY", None),
            size_mb=85.0,
            description="4-layer Conformer lightweight acoustic model for ultra-low latency edge devices",
            fallback_local_paths=["checkpoints/phono_v8_acoustic_tiny.pt"],
        ),
        "a2p_small_v8.0": WeightMetadata(
            key="a2p_small_v8.0",
            filename="phono_v8_acoustic_small.pt",
            part="a2p",
            size="small",
            version="v8.0",
            url=os.environ.get("PHONO_URL_A2P_SMALL", None),
            size_mb=195.0,
            description="6-layer Conformer mobile acoustic model optimized for Google Pixel 10 NPU",
            fallback_local_paths=["checkpoints/phono_v8_acoustic_small.pt"],
        ),

        # --- Models 2 & 3: Word Latent Encoder & Length-MoE Decoder (P2G) ---
        "p2g_base_v8.0": WeightMetadata(
            key="p2g_base_v8.0",
            filename="phono_v8_robust_p2g_base.pt",
            part="p2g",
            size="base",
            version="v8.0",
            url=os.environ.get("PHONO_URL_P2G_BASE", None),
            sha256="ade7b988659e984ce0307c91fc5b70c95648b4ccab5540ab63f6474c107c5ed4",
            size_mb=582.0,
            description="FastText Phoneme Word Encoder (Model 2) + 7 Overlapping Length Experts Word Decoder (Model 3)",
            fallback_local_paths=[
                "checkpoints/phono_v8_robust_p2g/best_robust_p2g.pt",
                "checkpoints/phono_v8_robust_p2g/latest_robust_p2g.pt",
                "checkpoints/phono_v8_robust_p2g_base.pt",
            ],
        ),
        "p2g_tiny_v8.0": WeightMetadata(
            key="p2g_tiny_v8.0",
            filename="phono_v8_robust_p2g_tiny.pt",
            part="p2g",
            size="tiny",
            version="v8.0",
            url=os.environ.get("PHONO_URL_P2G_TINY", None),
            size_mb=65.0,
            description="Lightweight 2-expert P2G linguistic decoder for on-device rollout",
            fallback_local_paths=["checkpoints/phono_v8_robust_p2g_tiny.pt"],
        ),
    }

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
        """
        Initialize WeightsManager.

        Args:
            cache_dir (str): Directory where downloaded checkpoints are cached.
        """
        self.cache_dir: str = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def set_url(self, key: str, url: str) -> None:
        """
        Registers or updates the GitHub Release asset download URL for a model checkpoint.

        Args:
            key (str): Registry key (e.g. 'a2p_base_v8.0', 'p2g_base_v8.0').
            url (str): Remote HTTPS download link.
        """
        if key in self.REGISTRY:
            self.REGISTRY[key].url = url
            print(f"[WeightsManager] Set URL for `{key}` -> {url}")
        else:
            raise KeyError(f"Unknown weight key: `{key}`. Valid keys: {list(self.REGISTRY.keys())}")

    def find_local_path(
        self,
        part: str = "a2p",
        size: str = "base",
        version: str = "v8.0",
    ) -> Optional[str]:
        """
        Searches for an existing checkpoint on disk among cache and known fallback paths.

        Args:
            part (str): 'a2p', 'p2g', or 'cnn'.
            size (str): 'tiny', 'small', 'base', 'large'.
            version (str): 'v8.0', 'latest'.

        Returns:
            Optional[str]: Absolute file path if found on disk, else None.
        """
        # Search direct key or by part attribute
        target_meta: Optional[WeightMetadata] = None
        exact_key = f"{part.lower()}_{size.lower()}_{version.lower()}"
        if exact_key in self.REGISTRY:
            target_meta = self.REGISTRY[exact_key]
        elif part.lower() in self.REGISTRY:
            target_meta = self.REGISTRY[part.lower()]
        else:
            for k, meta in self.REGISTRY.items():
                if meta.part.lower() == part.lower():
                    if size.lower() in meta.size.lower() or meta.size == "base":
                        target_meta = meta
                        break

        if target_meta is None:
            return None

        meta = target_meta

        # 1. Check primary cache location
        primary_cache_file = os.path.join(self.cache_dir, meta.filename)
        if os.path.isfile(primary_cache_file):
            return primary_cache_file

        # 2. Check known fallback project directories
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for rel_path in meta.fallback_local_paths:
            abs_path = os.path.join(project_root, rel_path)
            if os.path.isfile(abs_path):
                return abs_path

        return None

    def ensure_weights(
        self,
        part: str = "a2p",
        size: str = "base",
        version: str = "v8.0",
        destination_dir: Optional[str] = None,
    ) -> str:
        """
        Ensures checkpoint weights are present locally.
        If missing, automatically downloads them from the registered release URL.

        Args:
            part (str): Model part ('a2p', 'p2g', or 'cnn').
            size (str): Model size ('tiny', 'small', 'base', 'large').
            version (str): Model version ('v8.0', 'latest').
            destination_dir (Optional[str]): Target directory to store downloaded checkpoint.

        Returns:
            str: Absolute path to the validated local checkpoint.
        """
        local_path = self.find_local_path(part=part, size=size, version=version)
        if local_path and os.path.isfile(local_path):
            return local_path

        key = f"{part.lower()}_{size.lower()}_{version.lower()}"
        if key not in self.REGISTRY:
            matching = [k for k in self.REGISTRY if k.startswith(f"{part.lower()}_{size.lower()}")]
            if not matching:
                raise ValueError(f"[WeightsManager] Unknown weight spec: part={part}, size={size}, version={version}")
            key = matching[0]

        meta = self.REGISTRY[key]
        dest_dir = destination_dir or self.cache_dir
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, meta.filename)

        if not meta.url:
            raise FileNotFoundError(
                f"[WeightsManager] Weights `{meta.filename}` not found locally and no download URL is configured.\n"
                f"Please upload `{meta.filename}` ({meta.size_mb:.1f} MB) to your GitHub Release page\n"
                f"and provide the URL via: manager.set_url('{key}', '<release_url>')\n"
                f"or environment variable: PHONO_URL_{part.upper()}_{size.upper()}=<release_url>"
            )

        print(f"[WeightsManager] Downloading `{meta.filename}` ({meta.size_mb:.1f} MB) from:\n  {meta.url}")
        self._download_with_progress(meta.url, dest_path)

        # Verify SHA-256 if defined
        if meta.sha256:
            print("[WeightsManager] Verifying SHA-256 checksum...")
            file_hash = self._compute_sha256(dest_path)
            if file_hash != meta.sha256:
                os.remove(dest_path)
                raise ValueError(f"Checksum mismatch for `{dest_path}`! Expected {meta.sha256}, got {file_hash}")
            print("  - SHA-256 Checksum verified.")

        print(f"[WeightsManager] Successfully downloaded weights to: {dest_path}")
        return dest_path

    def _download_with_progress(self, url: str, destination: str) -> None:
        """Downloads a remote file displaying download progress."""
        def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded / total_size * 100.0)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb_down:.1f} / {mb_total:.1f} MB ({pct:.1f}%)")
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb_down:.1f} MB")
            sys.stdout.flush()

        temp_dest = destination + ".tmp"
        try:
            urllib.request.urlretrieve(url, temp_dest, reporthook=progress_callback)
            print("")
            os.replace(temp_dest, destination)
        except Exception as e:
            if os.path.exists(temp_dest):
                os.remove(temp_dest)
            raise RuntimeError(f"[WeightsManager] Failed to download from {url}: {e}")

    @staticmethod
    def _compute_sha256(file_path: str) -> str:
        """Computes SHA-256 hash of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha.update(chunk)
        return sha.hexdigest()

    def list_registry(self) -> None:
        """Prints a human-readable table of all configured models and local status."""
        print("=" * 85)
        print("                 AUDIOLEARN MODEL WEIGHTS & ASSETS REGISTRY                  ")
        print("=" * 85)
        print(f"{'Key':<16} | {'Part':<5} | {'Size':<6} | {'Size (MB)':<9} | {'Local Status':<14} | {'URL Status'}")
        print("-" * 85)
        for key, meta in self.REGISTRY.items():
            local = self.find_local_path(part=meta.part, size=meta.size, version=meta.version)
            loc_str = "FOUND" if local else "MISSING"
            url_str = "CONFIGURED" if meta.url else "NOT CONFIGURED"
            print(f"{key:<16} | {meta.part.upper():<5} | {meta.size:<6} | {meta.size_mb:>7.1f} MB | {loc_str:<14} | {url_str}")
        print("=" * 85)
