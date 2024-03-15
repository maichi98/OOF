from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
import torch
import ants


class Resample(MapTransform):

    def __init__(self, output_size, keys: KeysCollection):
        super().__init__(keys)
        self.output_size = output_size

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            x_final, y_final, z_final = self.output_size
            ants_img = ants.from_numpy(d[key])
            x_coef = ants_img.shape[0] / x_final
            y_coef = ants_img.shape[1] / y_final
            z_coef = ants_img.shape[2] / z_final
            # B-spline resampling now :
            finn = ants.resample_image(ants_img,
                                       resample_params=(x_coef, y_coef, z_coef),
                                       use_voxels=False,
                                       interp_type=1)
            d[key] = finn.numpy()
        return d


class Normalize(MapTransform):
    def __init__(self, max_dose, keys: KeysCollection):
        super().__init__(keys)
        self.max_dose = max_dose

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key] / self.max_dose
        return d


class Concatenate(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        input_mask_dose = torch.cat(tensors=[torch.from_numpy(d[key]).unsqueeze(0) for key in self.key_iterator(d)],
                                    dim=0)
        d_result = {"input_mask_dose": input_mask_dose}
        d_result.update(d)
        return d_result
