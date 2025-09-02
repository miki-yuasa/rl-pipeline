#!/usr/bin/env python3
"""
Test script to verify the refactored scaled_same_convolve function
works correctly and is more efficient.
"""

import time

import numpy as np
import pytest

from rl_pipeline.core.eval.stats import scaled_same_convolve


def original_scaled_same_convolve(xx, size):
    """Original implementation for comparison"""
    b = np.ones(size) / size
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = np.ceil(size / 2)

    # 補正部分
    xx_mean[0] *= size / n_conv
    for i in range(1, int(n_conv)):
        xx_mean[i] *= size / (i + n_conv)
        xx_mean[-i] *= size / (i + n_conv - (size % 2))
    # size%2は奇数偶数での違いに対応するため

    return xx_mean


@pytest.fixture
def test_data():
    """Fixture providing test data for the scaled convolution tests"""
    return [
        (np.random.rand(100), 5),
        (np.random.rand(100), 10),
        (np.random.rand(50), 7),
        (np.random.rand(200), 15),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3),
    ]


@pytest.mark.parametrize("window_size", [20, 50, 100])
def test_performance_comparison(window_size):
    """Test the performance improvement of the refactored function"""
    # Create larger test data for performance testing
    data = np.random.rand(10000)

    # Time original function
    start_time = time.time()
    for _ in range(100):  # Run multiple times for better timing
        original_scaled_same_convolve(data.copy(), window_size)
    original_time = time.time() - start_time

    # Time refactored function
    start_time = time.time()
    for _ in range(100):  # Run multiple times for better timing
        scaled_same_convolve(data.copy(), window_size)
    refactored_time = time.time() - start_time

    speedup = original_time / refactored_time

    # Print performance metrics
    print(f"\nWindow size: {window_size}")
    print(f"  Original time:   {original_time:.4f}s")
    print(f"  Refactored time: {refactored_time:.4f}s")
    print(f"  Speedup:         {speedup:.2f}x")

    # Assert that the refactored version is not significantly slower
    # Allow some tolerance for measurement variability
    assert refactored_time <= original_time * 1.1, (
        f"Refactored version is slower: {speedup:.2f}x"
    )


def test_correctness_with_fixture(test_data):
    """Test that the refactored function produces the same results as the original"""
    for i, (data, window_size) in enumerate(test_data):
        original_result = original_scaled_same_convolve(data.copy(), window_size)
        refactored_result = scaled_same_convolve(data.copy(), window_size)

        # Check if results are close (allowing for floating point precision)
        assert np.allclose(
            original_result, refactored_result, rtol=1e-10, atol=1e-10
        ), (
            f"Test case {i + 1} failed. Max difference: {np.max(np.abs(original_result - refactored_result))}"
        )


@pytest.mark.parametrize(
    "data,window_size",
    [
        (np.array([1, 2, 3, 4, 5]), 3),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 2),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 5),
        (np.random.rand(20), 7),
    ],
)
def test_correctness_parametrized(data, window_size):
    """Test correctness with parametrized test cases"""
    original_result = original_scaled_same_convolve(data.copy(), window_size)
    refactored_result = scaled_same_convolve(data.copy(), window_size)

    assert np.allclose(original_result, refactored_result, rtol=1e-10, atol=1e-10), (
        f"Results don't match. Max difference: {np.max(np.abs(original_result - refactored_result))}"
    )


def test_edge_cases():
    """Test edge cases for the scaled convolution function"""
    # Test with size 1 (should return original array)
    data = np.array([1, 2, 3, 4, 5])
    result = scaled_same_convolve(data, 1)
    expected = original_scaled_same_convolve(data, 1)
    assert np.allclose(result, expected)

    # Test with size equal to array length
    data = np.array([1, 2, 3, 4, 5])
    result = scaled_same_convolve(data, len(data))
    expected = original_scaled_same_convolve(data, len(data))
    assert np.allclose(result, expected)


def test_output_shape():
    """Test that output shape matches input shape"""
    data = np.random.rand(50)
    window_size = 7
    result = scaled_same_convolve(data, window_size)
    assert result.shape == data.shape, "Output shape should match input shape"


def test_output_type():
    """Test that output type is numpy array"""
    data = np.array([1, 2, 3, 4, 5])
    result = scaled_same_convolve(data, 3)
    assert isinstance(result, np.ndarray), "Output should be numpy array"
