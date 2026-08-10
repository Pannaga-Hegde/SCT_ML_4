"""Model complexity analysis module for GestureFlow.

Computes trainable/non-trainable parameter counts, model memory size (FP32),
estimated floating-point operations (FLOPs/MACs), and tensor dimensions.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.cnn import GestureCNN


@dataclass
class LayerComplexity:
    """Detailed complexity parameters for an individual neural network layer."""

    layer_name: str
    layer_type: str
    input_shape: List[int]
    output_shape: List[int]
    trainable_params: int
    non_trainable_params: int
    estimated_macs: int


@dataclass
class ModelComplexityReport:
    """Comprehensive neural network model complexity analysis report."""

    model_name: str
    input_shape: List[int]
    output_shape: List[int]
    trainable_params: int
    non_trainable_params: int
    total_params: int
    model_size_bytes_fp32: int
    model_size_mb_fp32: float
    total_macs: int
    total_flops: int
    layers: List[Dict]


class ModelComplexityAnalyzer:
    """Analyzes PyTorch model complexity including layer shapes, params, size, and FLOPs."""

    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 128, 128)) -> None:
        """Initialize ModelComplexityAnalyzer.

        Args:
            model: PyTorch neural network module.
            input_shape: Batch input tensor shape (B, C, H, W).
        """
        self.model = model
        self.input_shape = list(input_shape)

    def analyze(self) -> ModelComplexityReport:
        """Perform full layer-by-layer parameter and MACs analysis.

        Returns:
            ModelComplexityReport data structure.
        """
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        total_params = trainable_params + non_trainable_params

        # FP32 precision uses 4 bytes per parameter
        model_size_bytes = total_params * 4
        model_size_mb = round(model_size_bytes / (1024 * 1024), 4)

        layer_reports: List[LayerComplexity] = []
        total_macs = 0

        # Run dummy forward pass with hooks to compute exact layer activations & MACs
        device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device("cpu")
        dummy_input = torch.zeros(*self.input_shape, device=device)

        hooks = []
        layer_activations: Dict[str, Tuple[List[int], List[int]]] = {}

        def get_hook(name: str):
            def hook(module, input, output):
                in_shape = list(input[0].shape) if len(input) > 0 and isinstance(input[0], torch.Tensor) else []
                out_shape = list(output.shape) if isinstance(output, torch.Tensor) else []
                layer_activations[name] = (in_shape, out_shape)
            return hook

        for name, module in self.model.named_modules():
            if name != "" and not isinstance(module, nn.Sequential):
                hooks.append(module.register_forward_hook(get_hook(name)))

        self.model.eval()
        with torch.no_grad():
            output_tensor = self.model(dummy_input)

        for hook in hooks:
            hook.remove()

        output_shape = list(output_tensor.shape)

        for name, module in self.model.named_modules():
            if name == "" or isinstance(module, nn.Sequential):
                continue

            in_shape, out_shape = layer_activations.get(name, ([], []))
            
            p_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            p_non_trainable = sum(p.numel() for p in module.parameters(recurse=False) if not p.requires_grad)

            layer_macs = 0
            if isinstance(module, nn.Conv2d) and len(in_shape) == 4 and len(out_shape) == 4:
                # Conv2d MACs = out_h * out_w * out_channels * in_channels * k_h * k_w / groups
                b, c_in, h_in, w_in = in_shape
                _, c_out, h_out, w_out = out_shape
                k_h, k_w = module.kernel_size
                layer_macs = h_out * w_out * c_out * (c_in // module.groups) * k_h * k_w
                if module.bias is not None:
                    layer_macs += h_out * w_out * c_out
            elif isinstance(module, nn.Linear) and len(in_shape) >= 2 and len(out_shape) >= 2:
                # Linear MACs = in_features * out_features
                layer_macs = module.in_features * module.out_features
                if module.bias is not None:
                    layer_macs += module.out_features
            elif isinstance(module, nn.BatchNorm2d) and len(out_shape) == 4:
                # BatchNorm MACs = 2 * c * h * w
                _, c, h, w = out_shape
                layer_macs = 2 * c * h * w

            total_macs += layer_macs

            layer_reports.append(
                LayerComplexity(
                    layer_name=name,
                    layer_type=module.__class__.__name__,
                    input_shape=in_shape,
                    output_shape=out_shape,
                    trainable_params=p_trainable,
                    non_trainable_params=p_non_trainable,
                    estimated_macs=layer_macs,
                )
            )

        # Total FLOPs is roughly 2 * MACs (1 Multiply + 1 Accumulate per MAC)
        total_flops = 2 * total_macs

        return ModelComplexityReport(
            model_name=self.model.__class__.__name__,
            input_shape=self.input_shape,
            output_shape=output_shape,
            trainable_params=trainable_params,
            non_trainable_params=non_trainable_params,
            total_params=total_params,
            model_size_bytes_fp32=model_size_bytes,
            model_size_mb_fp32=model_size_mb,
            total_macs=total_macs,
            total_flops=total_flops,
            layers=[asdict(l) for l in layer_reports],
        )

    def save_json(self, output_path: Path) -> Path:
        """Analyze model complexity and save JSON report artifact.

        Args:
            output_path: Absolute target file path.

        Returns:
            Path to saved JSON file.
        """
        report = self.analyze()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=4)
        return output_path
