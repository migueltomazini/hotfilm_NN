#!/usr/bin/env python3
"""Integration test: Verify all modularized components work together."""

import sys
import os
sys.path.insert(0, '/home/mtomazini/Documents/GitHub/hotfilm_NN')

print("=" * 70)
print("INTEGRATION TEST: Modularized Components")
print("=" * 70 + "\n")

# Test 1: Import validation_metrics
print("✓ Test 1: Importing utils.validation_metrics...")
try:
    from utils import validation_metrics
    print("  SUCCESS: validation_metrics module imported\n")
except ImportError as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 2: Import hyperparameter_optimization
print("✓ Test 2: Importing utils.hyperparameter_optimization...")
try:
    from utils import hyperparameter_optimization
    print("  SUCCESS: hyperparameter_optimization module imported\n")
except ImportError as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 3: Import train_mlp (checks for circular dependencies)
print("✓ Test 3: Importing train_mlp (checks circular dependencies)...")
try:
    import train_mlp
    print("  SUCCESS: train_mlp imported without circular dependency errors\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 4: Verify MLP class is available
print("✓ Test 4: Verifying MLP class availability...")
try:
    from train_mlp import MLP
    print("  SUCCESS: MLP class accessible\n")
except ImportError as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 5: Verify VoltageVelocityDataset class is available
print("✓ Test 5: Verifying VoltageVelocityDataset class...")
try:
    from train_mlp import VoltageVelocityDataset
    print("  SUCCESS: VoltageVelocityDataset class accessible\n")
except ImportError as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 6: Import incremental_predict (should work with fixed imports)
print("✓ Test 6: Importing incremental_predict...")
try:
    import incremental_predict
    print("  SUCCESS: incremental_predict imported successfully\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 7: Import incremental_train
print("✓ Test 7: Importing incremental_train...")
try:
    import incremental_train
    print("  SUCCESS: incremental_train imported successfully\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 8: Verify validation_metrics functions exist
print("✓ Test 8: Verifying validation_metrics functions...")
expected_functions = [
    'count_trainable_parameters',
    'calculate_velocity_derivative_skewness',
    'calculate_real_time_ratio',
    'calculate_dissipation_series',
    'generate_1to1_scatterplot_data',
    'calculate_block_dissipation_continuity',
    'format_validation_report'
]

missing_functions = []
for func_name in expected_functions:
    if not hasattr(validation_metrics, func_name):
        missing_functions.append(func_name)

if missing_functions:
    print(f"  FAILED: Missing functions: {missing_functions}\n")
    sys.exit(1)
else:
    print(f"  SUCCESS: All {len(expected_functions)} functions present\n")

# Test 9: Quick function test
print("✓ Test 9: Quick functional test...")
try:
    import numpy as np
    
    # Test parameter counting
    params = validation_metrics.count_trainable_parameters(4, 56, 1, 3)
    assert params == 451, f"Expected 451 params, got {params}"
    
    # Test RTR calculation
    rtr = validation_metrics.calculate_real_time_ratio(5.0, 10.0)
    assert rtr == 0.5, f"Expected RTR=0.5, got {rtr}"
    
    # Test skewness on synthetic data
    synthetic_velocity = np.random.randn(1000, 3)
    skewness = validation_metrics.calculate_velocity_derivative_skewness(synthetic_velocity, 2000)
    assert 'u_longitudinal' in skewness, "Missing u_longitudinal in skewness output"
    
    print("  SUCCESS: All functional tests passed\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

# Test 10: Module interdependencies
print("✓ Test 10: Module interdependencies...")
try:
    # This checks that hyperparameter_optimization can be imported from train_mlp context
    from utils.hyperparameter_optimization import optimize_hyperparameters
    print("  SUCCESS: optimize_hyperparameters accessible\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

print("=" * 70)
print("✅ ALL INTEGRATION TESTS PASSED")
print("=" * 70)
print("\nModularization Summary:")
print("  • Circular imports resolved (lazy loading in hyperparameter_optimization)")
print("  • Command-line interfaces isolated (main() functions)")
print("  • Validation metrics integrated successfully")
print("  • All modules compile and import without errors")
print("  • Physics-informed metrics ready for production use")
print("\n✨ System is ready for deployment!")
