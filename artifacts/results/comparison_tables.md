# ONNX vs TVM Comparison

## Accuracy

| Metric | Value |
| --- | --- |
| ONNX accuracy | 0.9852 |
| TVM accuracy | 0.9852 |
| Prediction match rate | 1.0 |

## Performance

| Backend | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (samples/s) |
| --- | --- | --- | --- | --- |
| ONNX | 0.048 | 0.043 | 0.05 | 20874.7 |
| TVM | 2.349 | 0.179 | 14.023 | 425.8 |

## Model Size

| Representation | Size (KB) |
| --- | --- |
| ONNX (.onnx + external data) | 261.29 |
| TVM (lib + graph + params + relay) | 497.11 |

## Graph Structure

| Metric | Value |
| --- | --- |
| ONNX graph nodes | 14 |
| ONNX op-type counts | AveragePool=2, Concat=1, Conv=2, Gemm=3, Relu=4, Reshape=1, Shape=1 |
| Relay calls (post import) | 17 |
| Relay text lines | 25 |
| TVM compiled graph nodes | 20 |
| TVM graph arg nodes | 11 |
