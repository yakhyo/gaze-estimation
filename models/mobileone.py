# Modified by Yakhyokhuja Valikhujaev
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple

__all__ = [
    "mobileone_s0",
    "mobileone_s1",
    "mobileone_s2",
    "mobileone_s3",
    "mobileone_s4",
    "mobileone_s5",
    "reparameterize_model"
]


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """
        Construct a Squeeze and Excite Module.

        Args:
            in_channels (int): Number of input channels.
            rd_ratio (float): Input channel reduction ratio.
        """

        super().__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            inference_mode: bool = False,
            use_se: bool = False,
            num_conv_branches: int = 1
    ) -> None:
        """
        Construct a MobileOneBlock module.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the block.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple): Stride size.
            padding (int or tuple): Zero-padding size.
            dilation (int or tuple): Kernel dilation factor.
            groups (int): Group number.
            inference_mode (bool): If True, instantiates model in inference mode.
            use_se (bool): Whether to use SE-ReLU activations.
            num_conv_branches (int): Number of linear conv branches.
        """

        super().__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SqueezeExcitationBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ 
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` - https://arxiv.org/pdf/2101.03697.pdf. 
        We re-parameterize multi-branched architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fuse batchnorm layer with preceding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: The branch containing the convolutional and batchnorm layers to be fused.

        Returns:
            tuple: A tuple containing the kernel and bias after fusing batchnorm.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fuse batchnorm layer with preceding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: The branch containing the convolutional and batchnorm layers to be fused.

        Returns:
            tuple: A tuple containing the kernel and bias after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """
        Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size (int): Size of the convolution kernel.
            padding (int): Zero-padding size.

        Returns:
            nn.Sequential: A Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            'conv',
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False
            )
        )
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOne(nn.Module):
    """
    MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
            self,
            num_blocks_per_stage: List[int] = [2, 8, 10, 1],
            num_classes: int = 1000,
            width_multipliers: Optional[List[float]] = None,
            inference_mode: bool = False,
            use_se: bool = False,
            num_conv_branches: int = 1
    ) -> None:
        """
        Construct MobileOne model.

        Args:
            num_blocks_per_stage (list): List of number of blocks per stage.
            num_classes (int): Number of classes in the dataset.
            width_multipliers (list): List of width multipliers for blocks in a stage.
            inference_mode (bool): If True, instantiates model in inference mode.
            use_se (bool): Whether to use SE-ReLU activations.
            num_conv_branches (int): Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            inference_mode=self.inference_mode
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0)
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # yaw and pitch
        self.fc_yaw = nn.Linear(int(512 * width_multipliers[3]), num_classes)
        self.fc_pitch = nn.Linear(int(512 * width_multipliers[3]), num_classes)

        # self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int) -> nn.Sequential:
        """
        Build a stage of the MobileOne model.

        Args:
            planes (int): Number of output channels.
            num_blocks (int): Number of blocks in this stage.
            num_se_blocks (int): Number of SE blocks in this stage.

        Returns:
            nn.Sequential: A stage of the MobileOne model.
        """

        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches
                )
            )
            # Pointwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass . """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        # x = self.linear(x)

        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)

        return pitch, yaw


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """
    Re-parameterize the MobileOne model from a multi-branched structure (used in training)
    into a single branch for inference.

    Args:
        model (nn.Module): MobileOne model in training mode.

    Returns:
        nn.Module: MobileOne model re-parameterized for inference mode.
    """

    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


MOBILEONE_CONFIGS = {
    "mobileone_s0":
        {
            "weights": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0_unfused.pth.tar",
            "params":
                {
                    "width_multipliers": (0.75, 1.0, 1.0, 2.0),
                    "num_conv_branches": 4
                }
        },
    "mobileone_s1":
        {
            "weights": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s1_unfused.pth.tar",
            "params":
                {
                    "width_multipliers": (1.5, 1.5, 2.0, 2.5),
                }

        },
    "mobileone_s2":
        {
            "weights": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2_unfused.pth.tar",
            "params":
                {
                    "width_multipliers": (1.5, 2.0, 2.5, 4.0),
                }
        },
    "mobileone_s3":
        {
            "weights": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s3_unfused.pth.tar",
            "params":
                {
                    "width_multipliers": (2.0, 2.5, 3.0, 4.0),
                }
        },
    "mobileone_s4":
        {
            "weights": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s4_unfused.pth.tar",
            "params":
                {
                    "width_multipliers": (3.0, 3.5, 3.5, 4.0),
                    "use_se": True
                }
        }
}


def load_filtered_state_dict(model, state_dict):
    """Update the model's state dictionary with filtered parameters.

    Args:
        model: The model instance to update (must have `state_dict` and `load_state_dict` methods).
        state_dict: A dictionary of parameters to load into the model.
    """
    current_model_dict = model.state_dict()
    filtered_state_dict = {key: value for key, value in state_dict.items() if key in current_model_dict}
    current_model_dict.update(filtered_state_dict)
    model.load_state_dict(current_model_dict)


def create_mobileone_model(config, pretrained: bool = True, num_classes: int = 1000, inference_mode: bool = False) -> nn.Module:
    """
    Create a MobileOne model based on the specified architecture name.

    Args:
        config (dict): The configuration dictionary for the MobileOne model.
        pretrained (bool): If True, loads pre-trained weights for the specified architecture. Defaults to True.
        num_classes (int): Number of output classes for the model. Defaults to 1000.
        inference_mode (bool): If True, instantiates the model in inference mode. Defaults to False.

    Returns:
        nn.Module: The constructed MobileOne model.
    """
    weights = config["weights"]
    params = config["params"]

    model = MobileOne(num_classes=num_classes, inference_mode=inference_mode, **params)

    if pretrained:
        try:
            state_dict = torch.hub.load_state_dict_from_url(weights)
            load_filtered_state_dict(model, state_dict)
            logger.info("Pre-trained weights successfully loaded.")
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights. Exception: {e}")
            logger.info("Creating model without pre-trained weights.")
    else:
        logger.info("Creating model without pre-trained weights.")

    return model


def mobileone_s0(pretrained=True, num_classes=1000, inference_mode=False):
    return create_mobileone_model(MOBILEONE_CONFIGS['mobileone_s0'], pretrained, num_classes, inference_mode)


def mobileone_s1(pretrained=True, num_classes=1000, inference_mode=False):
    return create_mobileone_model(MOBILEONE_CONFIGS['mobileone_s1'], pretrained, num_classes, inference_mode)


def mobileone_s2(pretrained=True, num_classes=1000, inference_mode=False):
    return create_mobileone_model(MOBILEONE_CONFIGS['mobileone_s2'], pretrained, num_classes, inference_mode)


def mobileone_s3(pretrained=True, num_classes=1000, inference_mode=False):
    return create_mobileone_model(MOBILEONE_CONFIGS['mobileone_s3'], pretrained, num_classes, inference_mode)


def mobileone_s4(pretrained=True, num_classes=1000, inference_mode=False):
    return create_mobileone_model(MOBILEONE_CONFIGS['mobileone_s4'], pretrained, num_classes, inference_mode)


if __name__ == "__main__":
    model = mobileone_s2()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
