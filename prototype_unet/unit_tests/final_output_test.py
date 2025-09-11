from .. import model
from .. import configs
import torch

def test_final_output():
    in_channels = configs.C3 + configs.S1
    out_channels = 1

    final_output = model.FinalOutput(in_channels, out_channels)
    input_tensor = torch.randn(1, in_channels, 256, 256)
    output_tensor = final_output(input_tensor)

    assert output_tensor.shape == (1, out_channels, 256, 256), \
        f"Expected shape {(1, out_channels, 256, 256)}, but got {output_tensor.shape}"