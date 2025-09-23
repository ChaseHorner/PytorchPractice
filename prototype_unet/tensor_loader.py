import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import configs
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely import wkt


folder_path = r"Z:\prepped_data\S2fieldstacks20m_v0p2"
yield_path = r"Z:\ChaseSpring2025\processedData"
save_path = r"Z:\prepped_data\processed_tensors"

coord_dict = {
    "Ant_CN1NW": { "llc": (289980, 4408680), "urc": (295100, 4413800) },
    "Ant_CN4NW": { "llc": (285160, 4408820), "urc": (290280, 4413940) },
    "Yoos_Vrbas": { "llc": (331140, 4392580), "urc": (336260, 4397700) },
    "BFLP_RobbinsEast": { "llc": (341380, 4395260), "urc": (346500, 4400380) }}



def load_dataset(folder_path, save_path, dtype=torch.float32):
    for year in os.listdir(folder_path):
        year_path = os.path.join(folder_path, year)
        for field in os.listdir(year_path):
            field_path = os.path.join(year_path, field)
            if os.path.isdir(field_path):
                output_path = os.path.join(save_path, year, field)
                yield_mask = load_yield(field, year, output_path)
                load_field(field_path, output_path, yield_mask)


def load_field(field_path, output_path, yield_mask, dtype=torch.float32):
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

            if data_type == 'sentinel' and yield_mask is not None:
                final_tensor = torch.cat([final_tensor, yield_mask], dim=0)

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




def load_yield(field, year, output_path):

    yield_file = os.path.join(yield_path, field, year, f"{field}_{year}_harvest.csv")
    if not os.path.exists(yield_file):
        return None
    
    # Read the yield file (GeoDataFrame)
    df = gpd.read_file(yield_file)
    df['geometry'] = df['geometry'].apply(wkt.loads)  # convert WKT string to geometry
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:32614")
    
    # Compute target array size
    width  = int((coord_dict[field]['urc'][0] - coord_dict[field]['llc'][0]) / 20)
    height = int((coord_dict[field]['urc'][1] - coord_dict[field]['llc'][1]) / 20)

    # Define transform
    transform = from_origin(coord_dict[field]['llc'][0], coord_dict[field]['urc'][1], 20, 20)

    # Rasterize yield values into target grid
    raster = rasterize(
        ((geom, val) for geom, val in zip(gdf.geometry, gdf['adj_yield'].astype(np.float32))),
        out_shape=(height, width),
        transform=transform,
        fill=0.0,  # Fill background with 0
        dtype="float32"
    )

    # Convert to tensor (add channel dim)
    yield_tensor = torch.from_numpy(raster).unsqueeze(0)
    os.makedirs(output_path, exist_ok=True)
    save_file = os.path.join(output_path, f"target.pt")
    torch.save(yield_tensor, save_file)
    print(f"Saved {save_file}, shape {yield_tensor.shape}")

    yield_mask = (yield_tensor != 0).float()
    print(f"Returned yield mask, shape {yield_mask.shape}")

    return yield_mask


if __name__ == "__main__":
    load_dataset(folder_path, save_path)