from typing import Dict, Type, Any

from .unet import UNet
# from .resunet import ResUNet
# from .hybrid_transformer import HybridTransformer


MODEL_REGISTRY: Dict[str, Type] = {
    "unet": UNet,
    # "resunet": ResUNet,
    # "hybrid_transformer": HybridTransformer,
}


def create_model(name: str, **kwargs: Any):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name](**kwargs)
