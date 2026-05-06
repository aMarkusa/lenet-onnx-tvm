"""Quantize the LeNet ONNX model to INT8 (dynamic + static) and compare it
to an FP32 baseline on accuracy, file size, and CPU latency at batch 1.

The original lenet_mnist.onnx was exported with dynamo=True and the
graph it produces (with Shape/Concat around the flatten) confuses the
ORT quantizer's shape inference. So we re-export the trained checkpoint
with the legacy torch.onnx.export path and use that as the FP32 baseline.
The original ONNX is left alone so the FP32 vs TVM numbers in the report
stay valid.

Run:
    python quantize_int8_eval.py

Writes:
    artifacts/onnx/lenet_mnist_fp32_quantready.onnx
    artifacts/onnx/lenet_mnist_int8_dynamic.onnx
    artifacts/onnx/lenet_mnist_int8_static.onnx
    artifacts/results/quantization_int8.json
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "artifacts" / "data"
ONNX_DIR = PROJECT_ROOT / "artifacts" / "onnx"
CHECKPOINTS_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"

CHECKPOINT_PATH = CHECKPOINTS_DIR / "lenet_mnist.pt"
FP32_PATH = ONNX_DIR / "lenet_mnist_fp32_quantready.onnx"
DYNAMIC_INT8_PATH = ONNX_DIR / "lenet_mnist_int8_dynamic.onnx"
STATIC_INT8_PATH = ONNX_DIR / "lenet_mnist_int8_static.onnx"
RESULTS_PATH = RESULTS_DIR / "quantization_int8.json"

BATCH_SIZE = 1
TIMING_NUM_SAMPLES = 200
TIMING_NUM_WARMUP_RUNS = 10
CALIBRATION_SAMPLES = 200


class LeNetMNIST(nn.Module):
    # Must match the architecture in train_lenet_mnist.ipynb so we can
    # load the .pt checkpoint.
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def reexport_fp32_onnx():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run train_lenet_mnist.ipynb first."
        )
    model = LeNetMNIST()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        str(FP32_PATH),
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )


def make_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=transform)
    train_dataset = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return test_loader, train_loader


class MNISTCalibrationReader(CalibrationDataReader):
    def __init__(self, train_loader, input_name, n):
        self.input_name = input_name
        self._samples = []
        for inputs, _ in train_loader:
            self._samples.append({input_name: inputs.numpy().astype(np.float32)})
            if len(self._samples) >= n:
                break
        self._iter = None

    def get_next(self):
        if self._iter is None:
            self._iter = iter(self._samples)
        return next(self._iter, None)

    def rewind(self):
        self._iter = None


def evaluate_accuracy(model_path, test_loader):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        x = inputs.numpy().astype(np.float32)
        logits = session.run(None, {input_name: x})[0]
        preds = np.argmax(logits, axis=1)
        correct += int((preds == targets.numpy()).sum())
        total += int(targets.shape[0])
    return correct / total, total


def time_model(model_path, test_loader):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    timing_inputs = []
    for inputs, _ in test_loader:
        timing_inputs.append(inputs.numpy().astype(np.float32))
        if len(timing_inputs) >= TIMING_NUM_SAMPLES:
            break

    # warmup so first-run JIT / cache effects don't poison the timings
    for x in timing_inputs[:TIMING_NUM_WARMUP_RUNS]:
        session.run(None, {input_name: x})

    durations = []
    for x in timing_inputs:
        start = time.perf_counter()
        session.run(None, {input_name: x})
        durations.append(time.perf_counter() - start)

    arr = np.array(durations)
    total_duration = float(arr.sum())
    return {
        "num_runs": len(durations),
        "num_samples": len(durations) * BATCH_SIZE,
        "mean_latency_ms": float(arr.mean() * 1000),
        "median_latency_ms": float(np.median(arr) * 1000),
        "p95_latency_ms": float(np.percentile(arr, 95) * 1000),
        "throughput_samples_per_sec": float(len(durations) * BATCH_SIZE / total_duration),
    }


def file_size_bytes(path):
    return path.stat().st_size if path.is_file() else 0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Re-exporting FP32 ONNX from checkpoint via legacy exporter...")
    reexport_fp32_onnx()
    print(f"  -> {FP32_PATH.relative_to(PROJECT_ROOT)} ({file_size_bytes(FP32_PATH)} bytes)")

    test_loader, train_loader = make_loaders()

    fp32_session = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    input_name = fp32_session.get_inputs()[0].name
    print(f"input name: {input_name}")

    print("\nDynamic INT8 quantization (weights only)...")
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(DYNAMIC_INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"  -> {DYNAMIC_INT8_PATH.relative_to(PROJECT_ROOT)} ({file_size_bytes(DYNAMIC_INT8_PATH)} bytes)")

    # Static INT8 with calibration. Took a couple tries to get the recipe
    # right: QDQ + QInt8 leaves float Conv/Gemm in the graph because the
    # QDQ -> QLinear* fusion didn't happen, so it ends up *slower* than
    # FP32. QOperator + u8s8 (unsigned activations, signed weights) emits
    # QLinearConv / QGemm directly which is what ORT actually has fast
    # x86 kernels for. ORT also explicitly warns that QOperator + QInt8
    # is bad on x64.
    print(f"\nStatic INT8 quantization (QOperator/u8s8, {CALIBRATION_SAMPLES} calibration samples)...")
    calibration_reader = MNISTCalibrationReader(train_loader, input_name, n=CALIBRATION_SAMPLES)
    quantize_static(
        model_input=str(FP32_PATH),
        model_output=str(STATIC_INT8_PATH),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QOperator,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"  -> {STATIC_INT8_PATH.relative_to(PROJECT_ROOT)} ({file_size_bytes(STATIC_INT8_PATH)} bytes)")

    variants = {
        "fp32_baseline": FP32_PATH,
        "int8_dynamic": DYNAMIC_INT8_PATH,
        "int8_static": STATIC_INT8_PATH,
    }

    print("\nEvaluating accuracy and latency...")
    results = {
        "input_name": input_name,
        "calibration_samples": CALIBRATION_SAMPLES,
        "timing_num_samples": TIMING_NUM_SAMPLES,
        "timing_num_warmup_runs": TIMING_NUM_WARMUP_RUNS,
        "batch_size_eval": BATCH_SIZE,
        "variants": {},
    }

    for label, path in variants.items():
        size = file_size_bytes(path)
        accuracy, n_test = evaluate_accuracy(path, test_loader)
        timing = time_model(path, test_loader)
        results["variants"][label] = {
            "model_path": str(path.relative_to(PROJECT_ROOT)),
            "file_bytes": size,
            "num_test_samples": n_test,
            "accuracy": accuracy,
            "performance": timing,
        }
        print(f"  {label}: size={size} B, acc={accuracy:.4f}, "
              f"mean={timing['mean_latency_ms']:.3f} ms, "
              f"p95={timing['p95_latency_ms']:.3f} ms")

    fp32 = results["variants"]["fp32_baseline"]
    dyn = results["variants"]["int8_dynamic"]
    stat = results["variants"]["int8_static"]
    fp32_lat = fp32["performance"]["mean_latency_ms"]

    results["summary"] = {
        "fp32_to_dynamic_size_ratio": fp32["file_bytes"] / dyn["file_bytes"],
        "fp32_to_static_size_ratio": fp32["file_bytes"] / stat["file_bytes"],
        "fp32_to_dynamic_mean_speedup": fp32_lat / dyn["performance"]["mean_latency_ms"],
        "fp32_to_static_mean_speedup": fp32_lat / stat["performance"]["mean_latency_ms"],
        "accuracy_drop_dynamic": fp32["accuracy"] - dyn["accuracy"],
        "accuracy_drop_static": fp32["accuracy"] - stat["accuracy"],
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nWrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print("\nSummary:")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
