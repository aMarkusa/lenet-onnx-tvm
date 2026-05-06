"""Render the exported ONNX graph as a layered diagram.

Reads ``artifacts/onnx/lenet_mnist.onnx`` and writes
``artifacts/figures/onnx_graph.png``.

The figure is intended as a Netron-style overview of the operator topology of
the LeNet-style CNN: each ONNX node is shown as a labelled box, edges follow
tensor data flow between operators, and the layout is top-down by topological
depth. Initializer-only inputs (model parameters) are not drawn, to keep the
figure focused on the computational structure.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import onnx

PROJECT_ROOT = Path(__file__).resolve().parent
ONNX_PATH = PROJECT_ROOT / "artifacts" / "onnx" / "lenet_mnist.onnx"
FIGURE_PATH = PROJECT_ROOT / "artifacts" / "figures" / "onnx_graph.png"

OP_COLORS = {
    "Conv": "#4C72B0",
    "Gemm": "#4C72B0",
    "Relu": "#55A868",
    "AveragePool": "#DD8452",
    "MaxPool": "#DD8452",
    "Reshape": "#8172B2",
    "Concat": "#8172B2",
    "Shape": "#937860",
    "input": "#CCCCCC",
    "output": "#CCCCCC",
}


def load_graph(path: Path) -> onnx.GraphProto:
    model = onnx.load(str(path))
    return model.graph


def build_nx_graph(graph: onnx.GraphProto) -> tuple[nx.DiGraph, dict[str, str]]:
    """Build a DAG over operator nodes plus model input/output endpoints.

    Returns the DiGraph and a mapping from node id to its op-type label.
    """
    nx_graph = nx.DiGraph()
    label_map: dict[str, str] = {}

    # Names of tensors that originate from initializers (i.e. model
    # parameters). These are excluded so the diagram shows only the
    # computational data flow.
    initializer_names = {init.name for init in graph.initializer}

    # Add operator nodes.
    producer_of: dict[str, str] = {}
    for idx, node in enumerate(graph.node):
        op_label = node.op_type
        node_id = f"{op_label}_{idx}"
        nx_graph.add_node(node_id)
        label_map[node_id] = op_label
        for output_name in node.output:
            producer_of[output_name] = node_id

    # Add model input endpoints (excluding initializers).
    for value_info in graph.input:
        if value_info.name in initializer_names:
            continue
        node_id = f"input::{value_info.name}"
        nx_graph.add_node(node_id)
        label_map[node_id] = f"input\n({value_info.name})"
        producer_of[value_info.name] = node_id

    # Add model output endpoints.
    for value_info in graph.output:
        node_id = f"output::{value_info.name}"
        nx_graph.add_node(node_id)
        label_map[node_id] = f"output\n({value_info.name})"
        # Edges into the output endpoint will be added below as part of the
        # generic input handling for downstream nodes; we register the
        # endpoint here as a consumer rather than a producer.
        for node in graph.node:
            if value_info.name in node.output:
                nx_graph.add_edge(f"{node.op_type}_{list(graph.node).index(node)}", node_id)

    # Add edges between operator nodes via tensor producer/consumer relations.
    for idx, node in enumerate(graph.node):
        consumer_id = f"{node.op_type}_{idx}"
        for input_name in node.input:
            if input_name in initializer_names:
                continue
            producer_id = producer_of.get(input_name)
            if producer_id is None or producer_id == consumer_id:
                continue
            nx_graph.add_edge(producer_id, consumer_id)

    return nx_graph, label_map


def layered_layout(nx_graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Top-down layered layout that minimizes edge crossings via the
    barycenter heuristic over a small number of sweeps."""
    depth: dict[str, int] = {}
    for node_id in nx.topological_sort(nx_graph):
        preds = list(nx_graph.predecessors(node_id))
        depth[node_id] = 0 if not preds else max(depth[p] for p in preds) + 1

    layers: dict[int, list[str]] = defaultdict(list)
    for node_id, d in depth.items():
        layers[d].append(node_id)
    num_layers = max(layers.keys()) + 1

    # Initial in-layer ordering: alphabetical, deterministic.
    for d in range(num_layers):
        layers[d].sort()

    # Barycenter sweeps: alternate top-down / bottom-up, ordering each layer
    # by the mean position of neighbours in the adjacent already-fixed layer.
    def order_for(node_id: str, anchor_layer: list[str], use_predecessors: bool) -> float:
        anchor_pos = {n: i for i, n in enumerate(anchor_layer)}
        neighbours = (
            nx_graph.predecessors(node_id) if use_predecessors
            else nx_graph.successors(node_id)
        )
        positions = [anchor_pos[n] for n in neighbours if n in anchor_pos]
        return sum(positions) / len(positions) if positions else len(anchor_layer) / 2

    for _ in range(4):
        for d in range(1, num_layers):
            layers[d].sort(key=lambda n: order_for(n, layers[d - 1], use_predecessors=True))
        for d in range(num_layers - 2, -1, -1):
            layers[d].sort(key=lambda n: order_for(n, layers[d + 1], use_predecessors=False))

    pos: dict[str, tuple[float, float]] = {}
    max_layer_width = max(len(nodes) for nodes in layers.values())
    for layer_idx in range(num_layers):
        nodes = layers[layer_idx]
        n_nodes = len(nodes)
        for i, node_id in enumerate(nodes):
            x_offset = (i + 1) / (n_nodes + 1) * max_layer_width
            pos[node_id] = (x_offset, -layer_idx)
    return pos


def color_for(label: str) -> str:
    if label.startswith("input"):
        return OP_COLORS["input"]
    if label.startswith("output"):
        return OP_COLORS["output"]
    op_type = label.splitlines()[0]
    return OP_COLORS.get(op_type, "#BBBBBB")


def draw(nx_graph: nx.DiGraph, label_map: dict[str, str], path: Path) -> None:
    pos = layered_layout(nx_graph)
    fig, ax = plt.subplots(figsize=(7, 9))

    node_colors = [color_for(label_map[n]) for n in nx_graph.nodes()]
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_color=node_colors,
        node_shape="s",
        node_size=2400,
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        arrows=True,
        arrowsize=14,
        edge_color="#444444",
        node_size=2400,
        ax=ax,
    )
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        labels=label_map,
        font_size=8,
        font_color="white",
        ax=ax,
    )

    legend_entries = [
        ("Conv / Gemm", OP_COLORS["Conv"]),
        ("Relu", OP_COLORS["Relu"]),
        ("Pool", OP_COLORS["AveragePool"]),
        ("Reshape / Concat", OP_COLORS["Reshape"]),
        ("Shape", OP_COLORS["Shape"]),
        ("Model I/O", OP_COLORS["input"]),
    ]
    handles = [mpatches.Patch(color=color, label=label) for label, color in legend_entries]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    ax.set_title("ONNX graph of the LeNet-style CNN")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph = load_graph(ONNX_PATH)
    nx_graph, label_map = build_nx_graph(graph)
    draw(nx_graph, label_map, FIGURE_PATH)
    print(f"Wrote {FIGURE_PATH.relative_to(PROJECT_ROOT)} "
          f"({len(nx_graph.nodes)} nodes, {len(nx_graph.edges)} edges)")


if __name__ == "__main__":
    main()
