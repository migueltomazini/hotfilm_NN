#!/usr/bin/env python3
"""Simple test to verify all imports work without circular dependencies."""

import sys

try:
    import train_mlp
    print("✓ train_mlp imported successfully")
except Exception as e:
    print(f"✗ Error importing train_mlp: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from utils import hyperparameter_optimization
    print("✓ hyperparameter_optimization imported successfully")
except Exception as e:
    print(f"✗ Error importing hyperparameter_optimization: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import incremental_predict
    print("✓ incremental_predict imported successfully")
except Exception as e:
    print(f"✗ Error importing incremental_predict: {e}", file=sys.stderr)
    sys.exit(1)

print("\nAll imports successful!")
