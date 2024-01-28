import imageio.v2 as imageio
import torch

volum_path = "./volumetricData/lungscans"

volumetric_data = imageio.volread(volum_path, "DICOM")

volume_tensor = torch.from_numpy(volumetric_data)
volume_tensor = torch.unsqueeze(volume_tensor, 0 )
print(volume_tensor.shape)


