"""
INT8 quantization experiment.

Quantizes the LeNet model (dynamic + static) and compares against the
FP32 baseline on accuracy, file size, and latency.

NOTE: the original lenet_mnist.onnx (dynamo export) breaks ORT's shape
inference during quantization, so we re-export from the checkpoint with
the old-style torch.onnx.export. The dynamo-exported one is kept for the
TVM comparison.
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

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "lenet_mnist.pt"
DATA_ROOT = PROJECT_ROOT / "artifacts" / "data"
ONNX_DIR = PROJECT_ROOT / "artifacts" / "onnx"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"

FP32_PATH = ONNX_DIR / "lenet_mnist_fp32_quantready.onnx"
DYNAMIC_INT8_PATH = ONNX_DIR / "lenet_mnist_int8_dynamic.onnx"
STATIC_INT8_PATH = ONNX_DIR / "lenet_mnist_int8_static.onnx"
RESULTS_PATH = RESULTS_DIR / "quantization_int8.json"

BATCH_SIZE = 1
N_TIMING_SAMPLES = 200
N_WARMUP = 10
N_CALIBRATION = 200


# same arch as in train_lenet_mnist.ipynb, need it to load the checkpoint
class LeNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def reexport_fp32_onnx():
    """Re-export with legacy exporter (opset 13, no dynamic axes)."""
    assert CHECKPOINT_PATH.is_file(), f"missing checkpoint: {CHECKPOINT_PATH}"
    model = LeNetMNIST()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, 1, 28, 28), str(FP32_PATH),
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True, dynamo=False,
    )


def get_mnist_loaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tf)
    train_ds = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, train_loader


class MNISTCalibReader(CalibrationDataReader):
    """Feed calibration samples to quantize_static. Pre-loads N samples into memory."""

    def __init__(self, loader, input_name, n):
        self._data = []
        for imgs, _ in loader:
            self._data.append({input_name: imgs.numpy().astype(np.float32)})
            if len(self._data) >= n:
                break
        self._pos = 0

    def get_next(self):
        if self._pos >= len(self._data):
            return None
        out = self._data[self._pos]
        self._pos += 1
        return out

    def rewind(self):
        self._pos = 0


def eval_accuracy(onnx_path, test_loader):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    correct, total = 0, 0
    for imgs, labels in test_loader:
        out = sess.run(None, {name: imgs.numpy().astype(np.float32)})[0]
        correct += (out.argmax(1) == labels.numpy()).sum()
        total += len(labels)
    return correct / total, total


def bench_latency(onnx_path, test_loader):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    # grab a fixed set of inputs
    samples = []
    for imgs, _ in test_loader:
        samples.append(imgs.numpy().astype(np.float32))
        if len(samples) >= N_TIMING_SAMPLES:
            break

    # warmup
    for x in samples[:N_WARMUP]:
        sess.run(None, {name: x})

    times = []
    for x in samples:
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    return {
        "num_runs": len(times),
        "mean_latency_ms": float(arr.mean() * 1000),
        "median_latency_ms": float(np.median(arr) * 1000),
        "p95_latency_ms": float(np.percentile(arr, 95) * 1000),
        "throughput_samples_per_sec": float(len(times) / arr.sum()),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # step 1: get a quantizer-friendly FP32 onnx
    print("re-exporting fp32 onnx (legacy exporter)...")
    reexport_fp32_onnx()
    print(f"  {FP32_PATH.name}: {FP32_PATH.stat().st_size} bytes")

    test_loader, train_loader = get_mnist_loaders()
    input_name = ort.InferenceSession(
        str(FP32_PATH), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    # step 2: dynamic quantization (weights only, no calibration needed)
    print("\nrunning dynamic quantization...")
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(DYNAMIC_INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"  {DYNAMIC_INT8_PATH.name}: {DYNAMIC_INT8_PATH.stat().st_size} bytes")

    # step 3: static quantization with calibration data
    # I tried a few combos here. QDQ format with signed activations left
    # the Conv/Gemm as float ops (fusion didn't kick in) and ended up slower
    # than fp32. QOperator with uint8 activations + int8 weights produces
    # real QLinearConv/QGemm which is what ORT has actual x86 kernels for.
    print(f"\nrunning static quantization ({N_CALIBRATION} calibration samples)...")
    calib = MNISTCalibReader(train_loader, input_name, N_CALIBRATION)
    quantize_static(
        model_input=str(FP32_PATH),
        model_output=str(STATIC_INT8_PATH),
        calibration_data_reader=calib,
        quant_format=QuantFormat.QOperator,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"  {STATIC_INT8_PATH.name}: {STATIC_INT8_PATH.stat().st_size} bytes")

    # step 4: evaluate all three
    print("\nevaluating...")
    variants = {
        "fp32_baseline": FP32_PATH,
        "int8_dynamic": DYNAMIC_INT8_PATH,
        "int8_static": STATIC_INT8_PATH,
    }
    results = {
        "batch_size": BATCH_SIZE,
        "n_calibration": N_CALIBRATION,
        "n_timing_samples": N_TIMING_SAMPLES,
        "n_warmup": N_WARMUP,
        "variants": {},
    }
    for label, path in variants.items():
        acc, n = eval_accuracy(path, test_loader)
        perf = bench_latency(path, test_loader)
        results["variants"][label] = {
            "model_path": str(path.relative_to(PROJECT_ROOT)),
            "file_bytes": path.stat().st_size,
            "accuracy": acc,
            "n_test": n,
            "performance": perf,
        }
        print(f"  {label}: {path.stat().st_size}B  acc={acc:.4f}  "
              f"lat={perf['mean_latency_ms']:.3f}ms")

    # ratios
    fp32_v = results["variants"]["fp32_baseline"]
    dyn_v = results["variants"]["int8_dynamic"]
    stat_v = results["variants"]["int8_static"]
    results["summary"] = {
        "size_ratio_dynamic": fp32_v["file_bytes"] / dyn_v["file_bytes"],
        "size_ratio_static": fp32_v["file_bytes"] / stat_v["file_bytes"],
        "speedup_dynamic": fp32_v["performance"]["mean_latency_ms"] / dyn_v["performance"]["mean_latency_ms"],
        "speedup_static": fp32_v["performance"]["mean_latency_ms"] / stat_v["performance"]["mean_latency_ms"],
        "acc_drop_dynamic": fp32_v["accuracy"] - dyn_v["accuracy"],
        "acc_drop_static": fp32_v["accuracy"] - stat_v["accuracy"],
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
