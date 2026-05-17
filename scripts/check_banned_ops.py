"""
Check for banned ONNX operations in a model graph.

Banned ops break cross-platform export to TFLite/CoreML/NCNN and must not
appear in the inference graph. See PLAN.md "Banned ops in inference graph"
section for context and rationale.

This script is used by QA Gate 1 to verify untrained exports are
deployment-clean before any training spend.
"""

import argparse
import sys
from pathlib import Path

import onnx


def check_banned_ops(onnx_path: str) -> int:
    """
    Load an ONNX model and verify no banned op types appear in the graph.

    Args:
        onnx_path: Path to the .onnx file to check.

    Returns:
        0 if no banned ops found, 1 if any banned op is detected.
    """
    # Load the ONNX model
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        print(f"ERROR: Failed to load ONNX file: {e}", file=sys.stderr)
        return 1

    # Extract unique op_type values from graph nodes
    op_types = set()
    for node in model.graph.node:
        op_types.add(node.op_type)

    # Define banned operations that break cross-platform export
    banned_set = {"FFT", "STFT", "GridSample", "GroupNormalization", "If", "Loop", "Scan"}

    # Print all op types found (sorted)
    print("Op types found in graph:")
    for op_type in sorted(op_types):
        print(f"  {op_type}")

    # Check for banned operations
    detected_banned = op_types & banned_set
    if detected_banned:
        print(f"\nBANNED OPS DETECTED: {detected_banned}")
        return 1

    print("\nOK — no banned ops in graph")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check for banned ONNX op types in a model graph."
    )
    parser.add_argument(
        "onnx_path",
        help="Path to the .onnx file to check",
    )

    args = parser.parse_args()

    # Verify the file exists
    onnx_file = Path(args.onnx_path)
    if not onnx_file.exists():
        print(f"ERROR: File not found: {args.onnx_path}", file=sys.stderr)
        sys.exit(1)

    sys.exit(check_banned_ops(str(onnx_file)))


if __name__ == "__main__":
    main()
