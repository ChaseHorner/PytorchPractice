import torch
import time
import configs
from model import Unet
from dataGen import get_random_dataloader
import types

# Create a temporary "config module" dynamically
temp_configs = types.SimpleNamespace(**{k: getattr(configs, k) for k in dir(configs) if k.isupper()})

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Example input size, adjust to your data
BATCH_SIZE = 1
CHANNELS = configs.LIDAR_IN_CHANNELS  # or your first input channels
H, W = 5120, 5120  # typical UNet input size
train_loader = get_random_dataloader(num_samples=1, batch_size=1)

# Make a list of layers you want to sweep
layer_names = ['C1','C2','C3','C4','C5','C6','C7','C8']  # match your config variable names
base_channels = [getattr(configs, ln) for ln in layer_names]

# Define low/medium/high values per layer
def layer_options(base):
    low = max(1, base // 2)
    med = base
    high = base * 2
    return [low, med, high]

results = []

for i, ln in enumerate(layer_names):
    for val in layer_options(base_channels[i]):
        setattr(temp_configs, ln, val)
        
        # Build model with modified config
        model = Unet(config = temp_configs).to(device)
        
        model.eval()
        torch.cuda.empty_cache()
        
        # Warmup
        with torch.no_grad():
            lidar, sentinel, in_season, pre_season, target = next(iter(train_loader))
            model(lidar.to(device), sentinel.to(device), in_season.to(device), pre_season.to(device))
        
        # Measure forward pass
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        with torch.no_grad():
            _ = model(lidar.to(device), sentinel.to(device), in_season.to(device), pre_season.to(device))
        elapsed = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)  # in GB
        
        results.append({
            'layer': ln,
            'value': val,
            'forward_time_sec': elapsed,
            'peak_gpu_mem_GB': peak_mem
        })
        
        print(f"[{ln}={val}] Forward: {elapsed:.4f}s, Peak Mem: {peak_mem:.2f} GB")

# Optional: print nicely or save to CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('layer_sweep_results.csv', index=False)
print(df)
