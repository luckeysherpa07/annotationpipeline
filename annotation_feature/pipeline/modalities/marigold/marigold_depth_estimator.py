"""Marigold depth estimation using the official Hugging Face diffusers pipeline."""

import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


class MarigoldDepthEstimator:
    """Marigold depth estimation wrapper."""

    def __init__(self, model_name: str = "prs-eth/marigold-depth-v1-1", device: str = "auto"):
        self.model_name = model_name
        self.device = self._select_device(device)
        self.pipeline = None
        self.torch = None
        self._initialize_model()

    def _select_device(self, device: str) -> str:
        """Select the inference device."""
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _initialize_model(self) -> None:
        """Load the official Marigold depth pipeline."""
        try:
            from diffusers import MarigoldDepthPipeline
            import torch

            print(f"Loading Marigold model: {self.model_name}")
            print(f"Using device: {self.device}")

            self.torch = torch
            self.pipeline = MarigoldDepthPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            )
            self.pipeline.to(self.device)

            print("Loaded Marigold model successfully")
        except ImportError as e:
            raise ImportError(
                "Marigold dependencies not installed. "
                "Install with: pip install diffusers transformers accelerate safetensors torch torchvision pillow"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Marigold model: {e}") from e

    def estimate_depth(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Estimate a single depth map and normalize it for 16-bit PNG storage."""
        image_path = Path(image_path)

        if not image_path.exists():
            print(f"    ERROR: Image not found: {image_path}")
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            with self.torch.no_grad():
                result = self.pipeline(image)

            depth_map = np.asarray(result.prediction).squeeze()
            if depth_map.ndim != 2:
                print(
                    "    ERROR: Unexpected depth prediction shape for "
                    f"{image_path.name}: {depth_map.shape}"
                )
                return None

            max_value = float(depth_map.max()) if depth_map.size else 0.0
            if max_value <= 0:
                print(f"    ERROR: Empty depth prediction for {image_path.name}")
                return None

            return (depth_map / max_value * 65535).astype(np.uint16)
        except Exception as e:
            print(f"    ERROR: Depth estimation failed for {image_path.name}: {e}")
            return None

    def estimate_depth_batch(
        self,
        image_paths: list[Union[str, Path]],
        output_dir: Union[str, Path],
        save_format: str = "png",
    ) -> list[Path]:
        """Estimate and save depth maps for a batch of RGB images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, image_path in enumerate(image_paths, 1):
            image_path = Path(image_path)
            depth_map = self.estimate_depth(image_path)
            if depth_map is None:
                print(f"    Skipped: {image_path.name}")
                continue

            output_name = image_path.stem + f"_depth.{save_format}"
            output_path = output_dir / output_name

            if save_format == "png":
                write_ok = cv2.imwrite(str(output_path), depth_map)
                if not write_ok or not output_path.exists():
                    print(f"    ERROR: Failed to write depth PNG: {output_path}")
                    continue
            elif save_format == "npy":
                np.save(str(output_path), depth_map)
                if not output_path.exists():
                    print(f"    ERROR: Failed to write depth NPY: {output_path}")
                    continue
            else:
                raise ValueError(f"Unsupported format: {save_format}")

            saved_paths.append(output_path)
            print(f"    [{i}/{len(image_paths)}] Saved: {output_path}")

        return saved_paths


def get_depth_estimator(
    model_name: str = "prs-eth/marigold-depth-v1-1",
    device: str = "auto",
) -> MarigoldDepthEstimator:
    """Factory function for the Marigold depth estimator."""
    return MarigoldDepthEstimator(model_name=model_name, device=device)
