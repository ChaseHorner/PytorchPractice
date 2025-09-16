import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import configs


folder_path = r"Z:\prepped_data\S2fieldstacks20m_v0p2"
save_path = r"Z:\prepped_data\processed_tensors"

def load_dataset(folder_path, save_path, dtype=torch.float32):
    for year in os.listdir(folder_path):
        year_path = os.path.join(folder_path, year)
        for field in os.listdir(year_path):
            field_path = os.path.join(year_path, field)
            if os.path.isdir(field_path):
                output_path = os.path.join(save_path, year, field)
                load_field(field_path, output_path)


def load_field(field_path, output_path, dtype=torch.float32):
        for data_type in os.listdir(field_path):
            data_type_path = os.path.join(field_path, data_type)
            if os.path.isdir(data_type_path):
                data = os.listdir(data_type_path)
                data.sort()  # ensure consistent order
                tensors = []
                for file in data:
                    if file.endswith('.tif'):
                        file_path = os.path.join(data_type_path, file)
                        with rasterio.open(file_path) as src:
                            arr = src.read().astype(np.float32)
                            if arr.ndim == 2:
                                arr = arr[None, :, :]  # add channel dimension
                            tensor = torch.from_numpy(arr).type(dtype)
                            tensors.append(tensor)

                if not tensors:
                    raise ValueError(f"No .tif files found in {data_type_path}, tensors list is empty.")

                final_tensor = torch.cat(tensors, dim=0)

                 # Validate shape
                shape_dict = {"lidar" : configs.LIDAR_SIZE,
                              "sentinel" : configs.SEN_SIZE,}
                
                if data_type in shape_dict:
                    expected_shape = shape_dict[data_type]
                    if list(final_tensor.shape) != expected_shape:
                        raise ValueError(f"Shape mismatch for {data_type} in {field_path}. Expected {expected_shape}, got {list(final_tensor.shape)}")
                
                # Save tensor
                os.makedirs(output_path, exist_ok=True)
                save_file = os.path.join(output_path, f"{data_type}.pt")
                torch.save(final_tensor, save_file)
                print(f"Saved {save_file}, shape {final_tensor.shape}")

if __name__ == "__main__":
    load_dataset(folder_path, save_path)