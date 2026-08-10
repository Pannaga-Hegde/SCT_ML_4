"""Model summary generator module for GestureFlow.

Generates structured tabular text summaries of PyTorch architectures displaying
layer names, output shapes, and parameter counts per layer.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.cnn import GestureCNN


class ModelSummaryGenerator:
    """Generates human-readable tabular architecture summaries for PyTorch modules."""

    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 128, 128)) -> None:
        """Initialize ModelSummaryGenerator.

        Args:
            model: Neural network module.
            input_shape: Input batch tensor shape tuple.
        """
        self.model = model
        self.input_shape = input_shape

    def generate_summary_text(self) -> str:
        """Generate formatted tabular text string representation of model structure.

        Returns:
            Formatted multi-line text summary string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Model Summary: {self.model.__class__.__name__}")
        lines.append("=" * 80)
        lines.append(f"{'Layer (type/name)':<40} {'Output Shape':<22} {'Param #':<14}")
        lines.append("-" * 80)

        # Hook forward pass to capture exact output shapes
        device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device("cpu")
        dummy_input = torch.zeros(*self.input_shape, device=device)

        hooks = []
        output_shapes: Dict[str, List[int]] = {}

        def get_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    output_shapes[name] = list(output.shape)
            return hook

        for name, module in self.model.named_modules():
            if name != "" and not isinstance(module, nn.Sequential):
                hooks.append(module.register_forward_hook(get_hook(name)))

        self.model.eval()
        with torch.no_grad():
            self.model(dummy_input)

        for hook in hooks:
            hook.remove()

        total_params = 0
        trainable_params = 0

        for name, module in self.model.named_modules():
            if name == "" or isinstance(module, nn.Sequential):
                continue

            mod_params = sum(p.numel() for p in module.parameters(recurse=False))
            mod_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

            total_params += mod_params
            trainable_params += mod_trainable

            out_shape = str(output_shapes.get(name, "[]"))
            layer_label = f"{name} ({module.__class__.__name__})"
            if len(layer_label) > 38:
                layer_label = layer_label[:35] + "..."

            lines.append(f"{layer_label:<40} {out_shape:<22} {mod_params:<14,}")

        non_trainable_params = total_params - trainable_params
        model_size_mb = (total_params * 4) / (1024 * 1024)

        lines.append("=" * 80)
        lines.append(f"Total params: {total_params:,}")
        lines.append(f"Trainable params: {trainable_params:,}")
        lines.append(f"Non-trainable params: {non_trainable_params:,}")
        lines.append(f"Estimated FP32 Model Size: {model_size_mb:.3f} MB ({total_params * 4:,} bytes)")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_summary(self, output_path: Path) -> Path:
        """Generate and save text summary artifact to destination file path.

        Args:
            output_path: Target destination path.

        Returns:
            Path to saved summary text file.
        """
        summary_text = self.generate_summary_text()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        return output_path
