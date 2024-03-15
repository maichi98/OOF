from monai.networks.layers.factories import Norm
from monai.networks.nets import BasicUNet
from oof.exceptions import ModelException
from pathlib import Path
from torch import nn
import torch


def get_model(model_name: str) -> nn.Module:
    """
    get trained OOF prediction model based on model name
    :param model_name: str, Name of the model
    :return: nn.Module, returns trained model
    """

    if model_name == "baseline":
        return get_baseline_model()

    raise ModelException(f"{model_name} is not a supported model !")


def get_baseline_model() -> nn.Module:
    """
    The baseline model is a trained U-Net Architecture with 5 Blocks.
    the model's features are : features = (32, 64, 128, 256, 512, 32)
    It's the model developed as V1 by Nathan Benzazon
    :return: nn.Module, returns trained baseline model
    """

    # Build U-Net model :
    model = BasicUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
        norm=Norm.INSTANCE,
        upsample="deconv"
    )

    # Load the trained model's weights:
    path_model = Path(__file__).parent.parent / "models" / "baseline model.pth"
    if not path_model.exists():
        raise ModelException(f"the path to the baseline model does not exist !")

    model.load_state_dict(torch.load(path_model))

    return model
