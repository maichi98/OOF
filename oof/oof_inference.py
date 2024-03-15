from oof.oof_transforms import Resample, Normalize, Concatenate
from monai.transforms import LoadImaged, ToNumpyd, Compose
from oof.exceptions import OofDatasetException
from oof.oof_models import get_model
from pathlib import Path
from typing import Union
import numpy as np
import torch


def oof_predict(path_input_dose: Union[Path, str],
                path_input_mask: Union[Path, str],
                path_output_dose: Union[Path, str],
                model_name="baseline",
                ):
    """
    Predict the Out of field dose from an input mask and input dose,
        Save the output dose to path_output_dose
    :param path_input_dose: Union[Path, str] path to the numpy input dose (In-field dose)
    :param path_input_mask: Union[Path, str] path to the input mask (patient's mask or phantom)
    :param path_output_dose: Union[Path, str] path to save the out
    :param model_name: str, model name. Defaults to "baseline"

    """

    # -------------------------------------- Check the input and the output path: --------------------------------------

    # ----- Check dict_input compatibility : ---------------------------------------------------------------------------

    dict_input = {"input_dose": path_input_dose,
                  "input_mask": path_input_mask}

    for k, v in dict_input.items():

        if not isinstance(v, Union[Path, str]):
            raise OofDatasetException(f"path_{k} must be an str or Path object !")
        dict_input[k] = Path(v)
        if v.suffix != ".npy":
            raise OofDatasetException(f"{k} must be a numpy array !")
        if not v.exists():
            raise OofDatasetException(f"path_{k} does not exist !")

    # ----- Check path_output_dose compatibility : ---------------------------------------------------------------------

    if not isinstance(path_output_dose, Union[Path, str]):
        raise OofDatasetException(f"path_output_dose must be an str or Path object !")
    path_output_dose = Path(path_output_dose)

    if not path_output_dose.parent.exists():
        raise OofDatasetException(f"{path_output_dose.parent} does not exist !")
    # ------------------------------------------------------------------------------------------------------------------

    #  Set the Device :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the oof prediction model, in eval mode :
    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Apply preprocessing transforms to the dict_input :

    transforms = Compose([
        LoadImaged(keys=("input_dose", "input_mask", ), image_only=True),
        ToNumpyd(keys=("input_dose", "input_mask")),
        Normalize(max_dose=100.00, keys=("input_dose",)),
        Resample(output_size=[64, 64, 256], keys=("input_dose", "input_mask")),
        Concatenate(keys=("input_dose", "input_mask"))
    ])

    input_data = transforms(dict_input)

    # Apply OOF model to input_data :
    output_dose = model(input_data["input_mask_dose"].unsqueeze(0).to(device))
    output_dose = output_dose.squeeze().detach().cpu().numpy()

    # Add in_field dose :
    output_dose[input_data["input_dose"] > 0] = input_data["input_dose"][input_data["input_dose"] > 0]
    output_dose = output_dose * input_data["input_mask"] * 100.00

    # Save output_dose :
    np.save(path_output_dose, output_dose)
