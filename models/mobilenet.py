import torch
from torch import nn, Tensor
from torchvision.models import MobileNet_V2_Weights

from typing import Any, Callable, List, Optional

__all__ = ["mobilenet_v2"]


def _make_divisible(v: float, divisor: int = 8) -> int:
    """This function ensures that all layers have a channel number divisible by 8"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(torch.nn.Sequential):
    """Convolutional block, consists of nn.Conv2d, nn.BatchNorm2d, nn.ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional = None,
            groups: int = 1,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: bool = False,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.01)
        ]

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(
                    in_planes,
                    hidden_dim,
                    kernel_size=1,
                    activation_layer=nn.ReLU6
                )
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_planes),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_planes
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            dropout (float): The droupout probability

        """
        super().__init__()

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1,  activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        self.fc_yaw = nn.Linear(self.last_channel, num_classes)
        self.fc_pitch = nn.Linear(self.last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Original FC layer from MobileNet V2
        # x = self.classifier(x)

        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)

        return pitch, yaw


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


def mobilenet_v2(*, pretrained: bool = True, progress: bool = True, **kwargs: Any) -> MobileNetV2:

    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = MobileNetV2(**kwargs)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        load_filtered_state_dict(model, state_dict)

    return model
