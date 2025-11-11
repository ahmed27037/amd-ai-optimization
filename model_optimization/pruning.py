"""
Model Pruning

Provides structured and unstructured pruning for model compression
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class Pruner:
    """Base pruner class"""
    
    def __init__(self, pruning_ratio: float = 0.5):
        """
        Initialize pruner
        
        Args:
            pruning_ratio: Ratio of weights to prune (0.0 to 1.0)
        """
        self.pruning_ratio = pruning_ratio
        self.masks = {}
    
    def prune(self, model: nn.Module) -> nn.Module:
        """
        Prune model weights
        
        Args:
            model: PyTorch model
        
        Returns:
            Pruned model
        """
        raise NotImplementedError
    
    def get_pruning_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get pruning statistics
        
        Args:
            model: Pruned model
        
        Returns:
            Statistics dictionary
        """
        total_params = 0
        pruned_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                if name in self.masks:
                    pruned_params += (self.masks[name] == 0).sum().item()
        
        return {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'remaining_params': total_params - pruned_params,
            'pruning_ratio': pruned_params / total_params if total_params > 0 else 0,
        }


class StructuredPruner(Pruner):
    """
    Structured Pruning
    
    Prunes entire channels or filters to maintain hardware efficiency
    """
    
    def __init__(self, pruning_ratio: float = 0.5, granularity: str = "channel"):
        """
        Initialize structured pruner
        
        Args:
            pruning_ratio: Ratio of channels/filters to prune
            granularity: Pruning granularity ("channel", "filter")
        """
        super().__init__(pruning_ratio)
        self.granularity = granularity
    
    def prune(self, model: nn.Module) -> nn.Module:
        """
        Prune model using structured pruning
        
        Args:
            model: PyTorch model
        
        Returns:
            Pruned model
        """
        model.eval()
        
        # Calculate importance scores for each channel/filter
        importance_scores = self._calculate_importance(model)
        
        # Create masks
        self._create_masks(model, importance_scores)
        
        # Apply masks
        self._apply_masks(model)
        
        logger.info(f"Structured pruning applied: {self.pruning_ratio*100:.1f}% {self.granularity}s pruned")
        
        return model
    
    def _calculate_importance(self, model: nn.Module) -> Dict[str, np.ndarray]:
        """
        Calculate importance scores for channels/filters
        
        Uses L1 norm as importance metric
        """
        importance_scores = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights = module.weight.data.cpu().numpy()
                
                if isinstance(module, nn.Conv2d):
                    if self.granularity == "channel":
                        # Importance per output channel
                        scores = np.sum(np.abs(weights), axis=(1, 2, 3))
                    else:  # filter
                        scores = np.sum(np.abs(weights), axis=(0, 2, 3))
                else:  # Linear
                    scores = np.sum(np.abs(weights), axis=1)
                
                importance_scores[name] = scores
        
        return importance_scores
    
    def _create_masks(self, model: nn.Module, importance_scores: Dict[str, np.ndarray]) -> None:
        """Create pruning masks based on importance scores"""
        self.masks = {}
        
        for name, module in model.named_modules():
            if name in importance_scores:
                scores = importance_scores[name]
                num_to_prune = int(len(scores) * self.pruning_ratio)
                
                # Get indices of least important channels/filters
                indices_to_prune = np.argsort(scores)[:num_to_prune]
                
                # Create mask
                mask = np.ones(len(scores), dtype=bool)
                mask[indices_to_prune] = False
                
                self.masks[name] = torch.from_numpy(mask)
    
    def _apply_masks(self, model: nn.Module) -> None:
        """Apply pruning masks to model"""
        for name, module in model.named_modules():
            if name in self.masks and isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = self.masks[name].to(module.weight.device)
                
                if isinstance(module, nn.Conv2d):
                    if self.granularity == "channel":
                        # Prune output channels
                        module.weight.data = module.weight.data[mask]
                        if module.bias is not None:
                            module.bias.data = module.bias.data[mask]
                    else:
                        # Prune input channels
                        module.weight.data = module.weight.data[:, mask]
                else:  # Linear
                    module.weight.data = module.weight.data[mask]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[mask]


class UnstructuredPruner(Pruner):
    """
    Unstructured Pruning
    
    Prunes individual weights (less hardware-efficient but more flexible)
    """
    
    def prune(self, model: nn.Module) -> nn.Module:
        """
        Prune model using unstructured pruning
        
        Args:
            model: PyTorch model
        
        Returns:
            Pruned model
        """
        model.eval()
        
        # Calculate importance (magnitude-based)
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.cpu().numpy().flatten())
        
        all_weights = np.concatenate(all_weights)
        threshold = np.percentile(np.abs(all_weights), self.pruning_ratio * 100)
        
        # Create masks
        for name, param in model.named_parameters():
            if param.requires_grad:
                mask = torch.abs(param.data) > threshold
                self.masks[name] = mask
                param.data *= mask.float()
        
        logger.info(f"Unstructured pruning applied: {self.pruning_ratio*100:.1f}% weights pruned")
        
        return model

