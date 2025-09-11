import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_layer_speed_impact(csv_path):
    df = pd.read_csv(csv_path)
    
    layers = sorted(df['layer'].unique())
    
    # LINE PLOT
    plt.figure(figsize=(10, 6))
    for layer in layers:
        layer_df = df[df['layer'] == layer]
        plt.plot(layer_df['value'], layer_df['forward_time_sec'], marker='o', label=layer)
    plt.xscale('log')
    plt.xlabel('Number of Channels')
    plt.ylabel('Forward Pass Time (s)')
    plt.title('Forward Pass Time vs Channels by Layer')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('line')
    
    # HEATMAP
    # Pivot the table: layers as rows, channel values as columns
    heat_data = df.pivot(index='layer', columns='value', values='forward_time_sec')
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(heat_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Forward Pass Time (s)')
    
    plt.xticks(ticks=np.arange(len(heat_data.columns)), labels=heat_data.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(heat_data.index)), labels=heat_data.index)
    plt.xlabel('Channel Size')
    plt.ylabel('Layer')
    plt.title('Forward Pass Time Heatmap')
    plt.tight_layout()
    plt.savefig('heat')

# Example usage:
plot_layer_speed_impact('layer_sweep_results.csv')
