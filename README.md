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
analyze_results.py          read the results JSON, build the comparison tables and figures
make_graph_figure.py        helper for the operator/graph-structure figure
main.tex / sample.bib       the report
artifacts/                  trained model, ONNX file, results CSVs/JSON, figures
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

To build the report you'll need `main.tex` and `sample.bib` (the
bibliography file) in the same directory:

```bash
latexmk -pdf -bibtex- -biber main.tex
```

Overleaf works too; it picks up biber automatically because the report
uses biblatex with `style=ieee, sorting=none`.

## What you should see

The exact numbers from my run, copied from `artifacts/results/summary.txt`:

- ONNX accuracy = TVM accuracy = 0.9852 on the 10 000 MNIST test images,
  prediction match rate 1.0, max absolute logit diff ~6.7e-06.
- ONNX file: 261 KB. TVM artifacts total: 497 KB (the bulk of the extra
  is the compiled kernel `.so`/`.tar.so`).
- ONNX mean latency at batch 1: ~0.048 ms. TVM mean latency at batch 1:
  ~2.35 ms. ONNX Runtime is substantially faster here.
- Graph node counts: ONNX = 14, Relay calls after import = 17, compiled
  TVM graph = 20.

Your absolute timings will differ depending on CPU, but
the qualitative picture (TVM matches the predictions, isn't smaller, and
isn't faster on a generic CPU without tuning) should hold.

## Notes / known limitations

- Single-threaded, batch size 1, no autotuning. With AutoTVM/MetaSchedule
  or a more aggressive target (e.g. `llvm -mcpu=...`) the TVM number
  would change, possibly a lot. That's flagged as future work in the
  report.
- INT8 quantization is *not* part of the experiment; it's only mentioned
  in the background and discussion. The pipeline is FP32 end to end.
- The Netron screenshot at `artifacts/figures/onnx_graph.png` is a
  placeholder. The report references it, but if you regenerate it from
  scratch, open the ONNX file in Netron and re-export.

## Versions I used

The `requirements.txt` pins minimum versions. Specifically I had:

- torch 2.3.x, torchvision 0.18.x
- onnx 1.16, onnxruntime 1.18
- apache-tvm 0.16 (CPU build)
- numpy 1.26, pandas 2.2
