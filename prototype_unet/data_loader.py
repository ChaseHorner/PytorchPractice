import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import configs


TOP_DIR = "/path/to/geotiffs"

class GeoTiffDataset(Dataset):
    def __init__(self, folder_path, transform=None, dtype=torch.float32):
        """
        folder_path: directory with GeoTIFF files
        transform: optional torchvision transforms
        dtype: torch tensor type
        """
        self.fields = []

        for year in os.listdir(folder_path):
            year_path = os.path.join(folder_path, year)
            for field in os.listdir(year_path):
                field_path = os.path.join(folder_path, year, field)
                if os.path.isdir(field_path):
                    files = {'lidar': [], 'sentinel': [], 'in_season': [], 'pre_season': [], 'label': []}
                    for file in os.listdir(field_path):
                        if file.endswith('.tif') or file.endswith('.tiff'):
                            if "aspect" in file or "dem" in file or "slp" in file:
                                files['lidar'].append(os.path.join(field_path, file))
                            # elif "sentinel" in file:
                            else:
                                files['sentinel'].append(os.path.join(field_path, file))
                            # elif "in_season" in file:
                            #     files['in_season'].append(os.path.join(field_path, file))
                            # elif "pre_season" in file:
                            #     files['pre_season'].append(os.path.join(field_path, file))
                            # elif "label" in file:
                            #     files['label'].append(os.path.join(field_path, file))
                    self.fields.append(files)

        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        lidar_list, sentinel_list, in_season_list, pre_season_list, label_list = [], [], [], [], []

        for file_type, paths in self.fields[idx].items():
            paths.sort() # ensure consistent order, very important!

            for path in paths:
                with rasterio.open(path) as src:
                    arr = src.read().astype(np.float32)
                    if arr.ndim == 2:
                        arr = arr[None, :, :]  # add channel
                    tensor = torch.from_numpy(arr).type(self.dtype)

                if file_type == 'lidar':
                    lidar_list.append(tensor)
                elif file_type == 'sentinel':
                    sentinel_list.append(tensor)
                elif file_type == 'in_season':
                    in_season_list.append(tensor)
                elif file_type == 'pre_season':
                    pre_season_list.append(tensor)
                elif file_type == 'label':
                    label_list.append(tensor)


        lidar_tensor = torch.cat(lidar_list, dim=0) if lidar_list else torch.empty(0)
        sentinel_tensor = torch.cat(sentinel_list, dim=0) if sentinel_list else torch.empty(0)
        in_season_tensor = torch.cat(in_season_list, dim=0) if in_season_list else torch.empty(0)
        pre_season_tensor = torch.cat(pre_season_list, dim=0) if pre_season_list else torch.empty(0)
        label_tensor = torch.cat(label_list, dim=0) if label_list else torch.empty(0)

        return {
            'lidar': lidar_tensor,
            'sentinel': sentinel_tensor,
            'in_season': in_season_tensor,
            'pre_season': pre_season_tensor,
            'label': label_tensor
        }

# Example usage
dataset = GeoTiffDataset(TOP_DIR)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.BATCH_SIZE, shuffle=True)

for batch in dataloader:
    print(batch.shape)  # (B, bands, H, W)
