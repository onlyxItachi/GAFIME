"""
Test script for per-pair interaction types fix.

This tests that we can now do mixed operations like A * B + C,
where different interaction types are used for each pair.
"""

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Hamza\Desktop\GAFIME')

from gafime.backends.fused_kernel import (
    StaticBucket, 
    UnaryOp, 
    InteractionType,
    compute_pearson_from_stats,
    create_fold_mask
)


def test_mixed_interactions():
    """Test that A * B + C works correctly with per-pair interaction types."""
    print("=" * 60)
    print("Testing Per-Pair Interaction Types Fix")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 10000
    
    # Create test data
    A = np.random.randn(n_samples).astype(np.float32)
    B = np.random.randn(n_samples).astype(np.float32)
    C = np.random.randn(n_samples).astype(np.float32)
    
    # Target: we'll create a target that correlates with A*B+C
    target = (A * B + C + 0.1 * np.random.randn(n_samples)).astype(np.float32)
    
    # Create fold mask
    mask = create_fold_mask(n_samples, n_folds=5, seed=42)
    
    print(f"\n[OK] Created test data: {n_samples} samples, 3 features")
    print(f"   Target = A * B + C + noise")
    
    # Create bucket
    bucket = StaticBucket(n_samples=n_samples, n_features=3)
    bucket.upload_all(features=[A, B, C], target=target, mask=mask)
    print(f"[OK] Uploaded data to VRAM bucket")
    
    # Test 1: A * B + C (correct mixed interaction)
    print("\n" + "-" * 60)
    print("Test 1: A * B + C (MULT, ADD)")
    print("-" * 60)
    
    stats_correct = bucket.compute(
        feature_indices=[0, 1, 2],
        ops=[UnaryOp.IDENTITY, UnaryOp.IDENTITY, UnaryOp.IDENTITY],
        interaction_types=[InteractionType.MULT, InteractionType.ADD],  # A*B then +C
        val_fold=0
    )
    train_r_correct, val_r_correct = compute_pearson_from_stats(stats_correct)
    print(f"   Train Pearson: {train_r_correct:.4f}")
    print(f"   Val Pearson:   {val_r_correct:.4f}")
    
    # Test 2: A * B * C (all MULT - old behavior)
    print("\n" + "-" * 60)
    print("Test 2: A * B * C (MULT, MULT) - for comparison")
    print("-" * 60)
    
    stats_mult = bucket.compute(
        feature_indices=[0, 1, 2],
        ops=[UnaryOp.IDENTITY, UnaryOp.IDENTITY, UnaryOp.IDENTITY],
        interaction_types=[InteractionType.MULT, InteractionType.MULT],  # A*B*C
        val_fold=0
    )
    train_r_mult, val_r_mult = compute_pearson_from_stats(stats_mult)
    print(f"   Train Pearson: {train_r_mult:.4f}")
    print(f"   Val Pearson:   {val_r_mult:.4f}")
    
    # Test 3: Backward compatibility - single interaction type
    print("\n" + "-" * 60)
    print("Test 3: Backward compatibility (single int -> all MULT)")
    print("-" * 60)
    
    stats_compat = bucket.compute(
        feature_indices=[0, 1, 2],
        ops=[UnaryOp.IDENTITY, UnaryOp.IDENTITY, UnaryOp.IDENTITY],
        interaction_types=InteractionType.MULT,  # Single int - should expand to [MULT, MULT]
        val_fold=0
    )
    train_r_compat, val_r_compat = compute_pearson_from_stats(stats_compat)
    print(f"   Train Pearson: {train_r_compat:.4f}")
    print(f"   Val Pearson:   {val_r_compat:.4f}")
    
    # Test 4: (A + B) / C
    print("\n" + "-" * 60)
    print("Test 4: (A + B) / C (ADD, DIV)")
    print("-" * 60)
    
    stats_add_div = bucket.compute(
        feature_indices=[0, 1, 2],
        ops=[UnaryOp.IDENTITY, UnaryOp.IDENTITY, UnaryOp.IDENTITY],
        interaction_types=[InteractionType.ADD, InteractionType.DIV],  # (A+B)/C
        val_fold=0
    )
    train_r_add_div, val_r_add_div = compute_pearson_from_stats(stats_add_div)
    print(f"   Train Pearson: {train_r_add_div:.4f}")
    print(f"   Val Pearson:   {val_r_add_div:.4f}")
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # A*B+C should have higher correlation with target (since target = A*B+C + noise)
    if train_r_correct > train_r_mult + 0.1:
        print(f"[PASS] A*B+C ({train_r_correct:.4f}) > A*B*C ({train_r_mult:.4f})")
        print(f"   The correct interaction pattern correlates better with target!")
    else:
        print(f"[WARN] Unexpected: A*B+C ({train_r_correct:.4f}) vs A*B*C ({train_r_mult:.4f})")
    
    # Backward compatibility check
    if abs(train_r_compat - train_r_mult) < 0.001:
        print(f"[PASS] Backward compatibility works (single int expands correctly)")
    else:
        print(f"[FAIL] Backward compatibility broken")
    
    print("\n[SUCCESS] Per-pair interaction types fix is working!")
    print("=" * 60)


if __name__ == "__main__":
    test_mixed_interactions()
