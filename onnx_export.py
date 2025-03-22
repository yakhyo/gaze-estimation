import os
import argparse
import torch
from config import data_config
from utils.helpers import get_model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Gaze Estimation Model ONNX Export')

    parser.add_argument(
        '-w', '--weight',
        default='resnet34.pt',
        type=str,
        help='Trained state_dict file path to open'
    )
    parser.add_argument(
        '-n', '--model',
        type=str,
        default='resnet34',
        choices=['resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'mobileone_s0'],
        help='Backbone network architecture to use'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='gaze360',
        choices=list(data_config.keys()),
        help='Dataset name for bin configuration'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic batch size and input dimensions for ONNX export'
    )

    return parser.parse_args()


@torch.no_grad()
def onnx_export(params):
    # Get dataset config for bins
    if params.dataset not in data_config:
        raise KeyError(f"Unknown dataset: {params.dataset}. Available options: {list(data_config.keys())}")
    bins = data_config[params.dataset]['bins']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_model(params.model, bins, inference_mode=True)
    model.to(device)

    # Load weights
    state_dict = torch.load(params.weight, map_location=device)
    model.load_state_dict(state_dict)
    print("Gaze model loaded successfully!")

    # Eval mode
    model.eval()

    # Generate ONNX output filename
    fname = os.path.splitext(os.path.basename(params.weight))[0]
    onnx_model = f'{fname}_gaze.onnx'
    print(f"==> Exporting model to ONNX format at '{onnx_model}'")

    # Dummy input: RGB image, 448x448
    dummy_input = torch.randn(1, 3, 448, 448).to(device)

    # Handle dynamic axes
    dynamic_axes = None
    if params.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'pitch': {0: 'batch_size'},
            'yaw': {0: 'batch_size'}
        }
        print("Exporting model with dynamic input shapes.")
    else:
        print("Exporting model with fixed input size: (1, 3, 448, 448)")

    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pitch', 'yaw'],
        dynamic_axes=dynamic_axes
    )

    print(f"Model exported successfully to {onnx_model}")


if __name__ == '__main__':
    args = parse_arguments()
    onnx_export(args)
