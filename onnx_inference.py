# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import uniface
import argparse
import numpy as np
import onnxruntime as ort

from typing import Tuple

from utils.helpers import draw_bbox_gaze


class GazeEstimationONNX:
    """
    Gaze estimation using ONNXRuntime (logits to radian decoded).
    """

    def __init__(self, model_path: str, session: ort.InferenceSession = None) -> None:
        """Initializes the GazeEstimationONNX class.

        Args:
            model_path (str): Path to the ONNX model file.
            session (ort.InferenceSession, optional): ONNX Session. Defaults to None.

        Raises:
            AssertionError: If model_path is None and session is not provided.
        """
        self.session = session
        if self.session is None:
            assert model_path is not None, "Model path is required for the first time initialization."
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
            )

        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        self.input_shape = (448, 448)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape

        self.input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:][::-1])

        outputs = self.session.get_outputs()
        output_names = [output.name for output in outputs]

        self.output_names = output_names
        assert len(output_names) == 2, "Expected 2 output nodes, got {}".format(len(output_names))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)  # Resize to 448x448

        image = image.astype(np.float32) / 255.0

        mean = np.array(self.input_mean, dtype=np.float32)
        std = np.array(self.input_std, dtype=np.float32)
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)  # CHW → BCHW

        return image_batch

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def decode(self, pitch_logits: np.ndarray, yaw_logits: np.ndarray) -> Tuple[float, float]:
        pitch_probs = self.softmax(pitch_logits)
        yaw_probs = self.softmax(yaw_logits)

        pitch = np.sum(pitch_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset

        return np.radians(pitch[0]), np.radians(yaw[0])

    def estimate(self, face_image: np.ndarray) -> Tuple[float, float]:
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {"input": input_tensor})

        return self.decode(outputs[0], outputs[1])


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Estimation ONNX Inference")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video path or camera index (e.g., 0 for webcam)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output video (optional)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle numeric webcam index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Failed to open video source: {args.source}")

    # Initialize Gaze Estimation model
    engine = GazeEstimationONNX(model_path=args.model)
    detector = uniface.RetinaFace()

    # Optional output writer
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, _ = detector.detect(frame)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue

            pitch, yaw = engine.estimate(face_crop)
            draw_bbox_gaze(frame, bbox, pitch, yaw)

        if writer:
            writer.write(frame)

        cv2.imshow("Gaze Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
