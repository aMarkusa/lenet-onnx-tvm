# Special Project: ONNX vs TVM on a LeNet/MNIST baseline

This is the code and artifacts for my special project, *Evaluating Model
Compilation for TinyML: A Case Study Using ONNX and TVM*. The report itself
is `main.tex`.

The short version: I trained a small LeNet-style CNN on MNIST in PyTorch,
exported it to ONNX, then compiled the ONNX model with Apache TVM for a CPU
target (`llvm`). I then compared ONNX Runtime and TVM on accuracy, model
size, graph structure, and latency. Spoiler: TVM matched the accuracy
(good), but on this tiny network with a generic CPU target and no
autotuning, it was neither smaller nor faster than ONNX Runtime. The
project is mostly about *why* that happens, not about chasing a speedup.

## What's in here

```
train_lenet_mnist.ipynb     train, evaluate, export to ONNX, sanity-check ONNX vs PyTorch
compile_and_run_tvm.ipynb   import the ONNX model into Relay, compile, run, compare to ONNX Runtime
quantize_int8_eval.py       INT8 quantization (dynamic + static), eval accuracy/size/latency
analyze_results.py          read the results JSON, build the comparison tables and figures
make_graph_figure.py        helper for the operator/graph-structure figure
main.tex / sample.bib       the report
artifacts/                  trained model, ONNX files, results CSVs/JSON, figures
```

The `artifacts/data/` directory (the MNIST download) and the TVM compiled
binaries are git-ignored, since they're regenerated when you re-run the
notebooks.

## Reproducing the results

I used Python 3.11 on Linux. Other 3.10+ versions probably work but I
haven't tested them.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The TVM install (`apache-tvm`) is the one part that was annoying. The
prebuilt wheel works on Linux x86_64; on macOS / Windows you'll most
likely need to build TVM from source. If `pip install apache-tvm` fails,
see the official install guide: https://tvm.apache.org/docs/install/.

Then run the notebooks in order:

```bash
jupyter lab
```

1. `train_lenet_mnist.ipynb` trains for 3 epochs (CPU is fine, a couple
   of minutes), saves the checkpoint, exports `lenet_mnist.onnx`, and
   verifies that ONNX Runtime matches PyTorch on the test set.
2. `compile_and_run_tvm.ipynb` imports the ONNX model into TVM Relay,
   compiles it, runs the test set through the compiled module, and
   writes `artifacts/results/onnx_vs_tvm_verification.json` along with
   the Relay text dumps.

Once both notebooks have run, the analysis script produces the report
tables and plots:

```bash
python analyze_results.py
```

This writes the CSVs and the `summary.txt` under `artifacts/results/`,
and the bar charts under `artifacts/figures/`.

For the INT8 quantization experiment (separate from the TVM pipeline):

```bash
python quantize_int8_eval.py
```

This takes the trained model, converts it to reduced-precision (INT8)
variants, measures accuracy and speed for each, and saves the results
to `artifacts/results/quantization_int8.json`. It runs in under 10
seconds.

## What you should see

The exact numbers from my run, copied from `artifacts/results/summary.txt`:

- ONNX accuracy = TVM accuracy = 0.9852 on the 10 000 MNIST test images,
  prediction match rate 1.0, max absolute logit diff ~6.7e-06.
- ONNX model file: 261 KB. TVM output: 497 KB total (bigger because it
  includes a compiled library that ONNX doesn't need).
- ONNX mean latency at batch 1: ~0.048 ms. TVM mean latency: ~2.35 ms.
  ONNX Runtime is substantially faster here because it already has
  hand-tuned kernels for this CPU, and TVM was run without any tuning.
- Graph node counts: ONNX = 14, Relay (TVM's intermediate form) = 17,
  final compiled TVM graph = 20. More nodes doesn't mean more work; the
  report explains where the extra ones come from.

INT8 quantization results (from `quantize_int8_eval.py`). The idea here
is to shrink the model by storing weights (and optionally activations) as
8-bit integers instead of 32-bit floats:

- FP32 baseline: 244 KB, accuracy 0.9852, latency ~0.031 ms.
- INT8 dynamic: 70 KB (~3.5x smaller), accuracy 0.9855, latency ~0.069 ms.
  "Dynamic" means the weights are stored as int8 ahead of time, but
  activations are quantized on the fly at each inference call.
- INT8 static: 69 KB (~3.5x smaller), accuracy 0.9851, latency ~0.051 ms.
  "Static" means both weights and activations use pre-computed int8
  scales, decided by running 200 training images through the model
  as calibration beforehand.
- The model gets ~3.5x smaller with no accuracy loss. But neither
  variant is actually faster than FP32 at batch size 1. The network is
  so small that the overhead of converting values between int8 and float
  at layer boundaries costs more time than the cheaper integer math
  saves. Same story as the TVM result: the workload is too tiny for the
  optimization to pay off on this hardware.

Your absolute timings will differ depending on CPU, but
the qualitative picture should hold.

## Notes / known limitations

- All benchmarks are single-threaded at batch size 1 (one image at a
  time), and TVM was not autotuned for this specific CPU. With tuning
  or larger batches the TVM and INT8 numbers would likely improve, but
  that's left as future work in the report.
- The Netron screenshot at `artifacts/figures/onnx_graph.png` is a
  placeholder. The report references it, but if you regenerate it from
  scratch, open the ONNX file in Netron and re-export.

## Versions I used

The `requirements.txt` pins minimum versions. Specifically I had:

- torch 2.3.x, torchvision 0.18.x
- onnx 1.16, onnxruntime 1.18
- apache-tvm 0.16 (CPU build)
- numpy 1.26, pandas 2.2
