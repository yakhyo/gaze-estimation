# MobileGaze: Pre-trained mobile nets for Gaze-Estimation

![Downloads](https://img.shields.io/github/downloads/yakhyo/gaze-estimation/total)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/gaze-estimation)](https://github.com/yakhyo/gaze-estimation/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/gaze-estimation)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14257640.svg)](https://doi.org/10.5281/zenodo.14257640)

<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5> 
-->

<!--
<div align="center">
  <img src="assets/out_video.gif">
</div>
-->

<!-- <video controls autoplay loop src="https://github.com/user-attachments/assets/a3af56a9-25af-4827-b716-27f610def59a" muted="false" width="100%"></video> -->
<div align="center">
 <img src="assets/out_gif.gif" width="100%">
 <p>
 Video by Yan Krukau: https://www.pexels.com/video/male-teacher-with-his-students-8617126/
 </p>
</div>

This project aims to perform gaze estimation using several deep learning models like ResNet, MobileNet v2, and MobileOne. It supports both classification and regression for predicting gaze direction. Built on top of [L2CS-Net](https://github.com/Ahmednull/L2CS-Net), the project includes additional pre-trained models and refined code for better performance and flexibility.

## Features

- [x] **ONNX Inference**: Export pytorch weights to ONNX and ONNX runtime inference.
- [x] **ResNet**: [Deep Residual Networks](https://arxiv.org/abs/1512.03385) - Enables deeper networks with better accuracy through residual learning.
- [x] **MobileNet v2**: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) - Efficient model for mobile applications, balancing performance and computational cost.
- [x] **MobileOne (s0-s4)**: [An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040) - Achieves near-instant inference times, ideal for real-time mobile applications.
- [x] **Face Detection**: [uniface](https://github.com/yakhyo/uniface) - **Uniface** face detection library uses RetinaFace model.

> [!NOTE]  
> All models are trained only on **Gaze360** dataset.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yakhyo/gaze-estimation.git
cd gaze-estimation
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download weight files:

   a) Download weights from the following links:

   | Model        | PyTorch Weights                                                                                             | ONNX Weights                                                                                                        | Size    | Epochs | MAE*  |
   | ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------- | ------ | ----- |
   | ResNet-18    | [resnet18.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet18.pt)               | [resnet18_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet18_gaze.onnx)         | 43 MB   | 200    | 12.84 |
   | ResNet-34    | [resnet34.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet34.pt)               | [resnet34_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet34_gaze.onnx)         | 81.6 MB | 200    | 11.33 |
   | ResNet-50    | [resnet50.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet50.pt)               | [resnet50_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet50_gaze.onnx)         | 91.3 MB | 200    | 11.34 |
   | MobileNet V2 | [mobilenetv2.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobilenetv2.pt)         | [mobilenetv2_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobilenetv2_gaze.onnx)   | 9.59 MB | 200    | 13.07 |
   | MobileOne S0 | [mobileone_s0_fused.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobileone_s0.pt) | [mobileone_s0_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobileone_s0_gaze.onnx) | 4.8 MB  | 200    | 12.58 |
   | MobileOne S1 | [not available](#)                                                                                          | [not available](#)                                                                                                  | xx MB   | 200    | \*    |
   | MobileOne S2 | [not available](#)                                                                                          | [not available](#)                                                                                                  | xx MB   | 200    | \*    |
   | MobileOne S3 | [not available](#)                                                                                          | [not available](#)                                                                                                  | xx MB   | 200    | \*    |
   | MobileOne S4 | [not availablet](#)                                                                                         | [not available](#)                                                                                                  | xx MB   | 200    | \*    |

   '\*' - soon will be uploaded (due to limited computing resources I cannot publish rest of the weights, but you still can train them with given code).
   
   *MAE (Mean Absolute Error) - lower values indicate better accuracy in degrees.

   b) Run the command below to download weights to the `weights` directory (Linux):

   ```bash
   # Download specific model weights
   sh download.sh [model_name]
   # Available models: resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0
   
   # Example:
   sh download.sh resnet18
   ```

## Usage

### Datasets

**Note**: Datasets must be downloaded separately and organized as shown below.

Dataset folder structure:

```
data/
├── Gaze360/
│   ├── Image/
│   └── Label/
└── MPIIFaceGaze/
    ├── Image/
    └── Label/
```

**Gaze360**

- Link to download dataset: https://gaze360.csail.mit.edu/download.php
- Data pre-processing code: https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#gaze360

**MPIIGaze**

- Link to download dataset: [download page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)
- Data pre-processing code: https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#mpiifacegaze

### Training

```bash
python main.py --data [dataset_path] --dataset [dataset_name] --arch [architecture_name]
```

`main.py` arguments:

```
usage: main.py [-h] [--data DATA] [--dataset DATASET] [--output OUTPUT] [--checkpoint CHECKPOINT] [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] [--arch ARCH] [--alpha ALPHA] [--lr LR] [--num-workers NUM_WORKERS]

Gaze estimation training.

options:
  -h, --help            show this help message and exit
  --data DATA           Directory path for gaze images.
  --dataset DATASET     Dataset name, available `gaze360`, `mpiigaze`.
  --output OUTPUT       Path of output models.
  --checkpoint CHECKPOINT
                        Path to checkpoint for resuming training.
  --num-epochs NUM_EPOCHS
                        Maximum number of training epochs.
  --batch-size BATCH_SIZE
                        Batch size.
  --arch ARCH           Network architecture, currently available: resnet18/34/50, mobilenetv2, mobileone_s0-s4.
  --alpha ALPHA         Regression loss coefficient.
  --lr LR               Base learning rate.
  --num-workers NUM_WORKERS
                        Number of workers for data loading.
```

### Evaluation

```bash
python evaluate.py --data [dataset_path] --dataset [dataset_name] --weight [weight_path] --arch [architecture_name]
```

`evaluate.py` arguments:

```
usage: evaluate.py [-h] [--data DATA] [--dataset DATASET] [--weights WEIGHTS] [--batch-size BATCH_SIZE] [--arch ARCH] [--num-workers NUM_WORKERS]

Gaze estimation evaluation.

options:
  -h, --help            show this help message and exit
  --data DATA           Directory path for gaze images.
  --dataset DATASET     Dataset name, available `gaze360`, `mpiigaze`
  --weights WEIGHTS     Path to model weight for evaluation.
  --batch-size BATCH_SIZE
                        Batch size.
  --arch ARCH           Network architecture, currently available: resnet18/34/50, mobilenetv2, mobileone_s0-s4.
  --num-workers NUM_WORKERS
                        Number of workers for data loading.
```

### Inference

```bash
# Run inference on webcam (camera index 0)
python inference.py --model resnet18 --weight weights/resnet18.pt --view --source 0

# Run inference on video file
python inference.py --model [model_name] --weight [model_weight_path] --view --source [source_video] --output [output_file] --dataset [dataset_name]
```

`inference.py` arguments:

```
usage: inference.py [-h] [--model MODEL] [--weight WEIGHT] [--view] [--source SOURCE] [--output OUTPUT] [--dataset DATASET]

Gaze estimation inference

options:
  -h, --help         show this help message and exit
  --model MODEL      Model name, default `resnet18`
  --weight WEIGHT    Path to gaze esimation model weights
  --view             Display the inference results
  --source SOURCE    Path to source video file or camera index
  --output OUTPUT    Path to save output file
  --dataset DATASET  Dataset name to get dataset related configs
```

### ONNX Export and Inference

**Export to ONNX**

```bash
python onnx_export.py --weight [model_path] --model [model_name] --dynamic
```

`onnx_export.py` arguments:

```
usage: onnx_export.py [-h] [-w WEIGHT] [-n {resnet18,resnet34,resnet50,mobilenetv2,mobileone_s0}] [-d {gaze360}] [--dynamic]

Gaze Estimation Model ONNX Export

options:
  -h, --help            show this help message and exit
  -w WEIGHT, --weight WEIGHT
                        Trained state_dict file path to open
  -n {resnet18,resnet34,resnet50,mobilenetv2,mobileone_s0}, --model {resnet18,resnet34,resnet50,mobilenetv2,mobileone_s0}
                        Backbone network architecture to use
  -d {gaze360,mpiigaze}, --dataset {gaze360,mpiigaze}
                        Dataset name for bin configuration
  --dynamic             Enable dynamic batch size and input dimensions for ONNX export
```

**ONNX Inference**

```bash
python onnx_inference.py --source [source video / webcam index] --model [onnx model path] --output [path to save video]
```

`onnx_inference.py` arguments:

```
usage: onnx_inference.py [-h] --source SOURCE --model MODEL [--output OUTPUT]

Gaze Estimation ONNX Inference

options:
  -h, --help       show this help message and exit
  --source SOURCE  Video path or camera index (e.g., 0 for webcam)
  --model MODEL    Path to ONNX model
  --output OUTPUT  Path to save output video (optional)
```

## Citation

If you use this work in your research, please cite it as:

Valikhujaev, Y. (2024). MobileGaze: Pre-trained mobile nets for Gaze-Estimation. Zenodo. [https://doi.org/10.5281/zenodo.14257640](https://doi.org/10.5281/zenodo.14257640)

Alternatively, in BibTeX format:

```bibtex
@misc{valikhujaev2024mobilegaze,
  author       = {Valikhujaev, Y.},
  title        = {MobileGaze: Pre-trained mobile nets for Gaze-Estimation},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14257640},
  url          = {https://doi.org/10.5281/zenodo.14257640}
}
```

## Reference

1. This project is built on top of [L2CS-Net](https://github.com/Ahmednull/L2CS-Net). Most of the code parts have been re-written for reproducibility and adaptability. Several additional backbones are provided with pre-trained weights.
2. https://github.com/apple/ml-mobileone
3. [uniface](https://github.com/yakhyo/uniface) - face detection library used for inference in `detect.py`.

<!--
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yakhyo/gaze-estimation&type=Date)](https://star-history.com/#yakhyo/gaze-estimation&Date)
-->
