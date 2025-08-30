# Computational Complexity

This page analyzes the computational complexity of the Barnes-Hut algorithm and discusses the accuracy-performance tradeoffs.

## Time Complexity

### Tree Construction: O(N log N)

Building the spatial tree requires:

1. **Sorting particles**: O(N log N) using Morton ordering
2. **Tree assembly**: O(N) to build the tree structure
3. **Computing node properties**: O(N) to calculate centers of mass

The dominating factor is the sorting step, giving overall O(N log N) complexity.

### Force Evaluation: O(M log N)

For M target particles querying forces from N source particles:

- **Tree traversal**: Each particle visits O(log N) nodes on average
- **Force calculation**: O(1) per node interaction
- **Total**: O(M log N)

When M = N (self-force calculation), this becomes O(N log N).

### Comparison with Direct Methods

| Method | Time Complexity | Accuracy |
|--------|----------------|----------|
| Direct (brute force) | O(N²) | Exact |
| Barnes-Hut | O(N log N) | Approximate |
| Fast Multipole Method | O(N) | Approximate |

## Space Complexity

### Memory Usage: O(N)

- **Particle storage**: 3N floats for positions (2N in 2D)
- **Tree nodes**: ~4N/3 nodes in worst case (complete tree)
- **Node properties**: Center of mass, total mass per node
- **Total**: O(N) memory usage

### Cache Efficiency

The tree structure is optimized for cache performance:

- Breadth-first node ordering improves locality
- Contiguous arrays reduce memory fragmentation
- Vectorized operations maximize throughput

## Accuracy vs Performance

### The θ Parameter

The accuracy-speed tradeoff is controlled by the θ (theta) parameter:

$$\text{Error} \propto \theta^2$$

$$\text{Speed} \propto \frac{1}{\log(1/\theta)}$$

| θ Value | Relative Error | Relative Speed | Use Case |
|---------|---------------|----------------|----------|
| 0.0 | 0% (exact) | 1× (slowest) | Validation |
| 0.1 | ~1% | 2-3× | High accuracy |
| 0.5 | ~25% | 5-10× | Balanced (recommended) |
| 1.0 | ~100% | 10-20× | Fast approximation |
| 2.0 | ~400% | 20-50× | Very rough estimate |

### Error Analysis

The error in Barnes-Hut comes from two sources:

1. **Approximation error**: Treating distant groups as point masses
2. **Truncation error**: Using first-order multipole expansion

The total relative error is approximately:

$$\epsilon_{rel} \approx \theta^2 \left(1 + \frac{s}{2d}\right)$$

Where s/d is the size-to-distance ratio of the farthest node used in approximation.

### Adaptive Accuracy

For applications requiring variable accuracy:

```python
# High accuracy for nearby interactions
forces_precise = bhut.force(positions, masses, theta=0.1)

# Lower accuracy for distant background
forces_approx = bhut.force(positions, masses, theta=1.0)
```

## Scaling Analysis

### Strong Scaling (Fixed Problem Size)

Performance with increasing number of processors:

- **Ideal speedup**: Limited by O(log N) tree traversal
- **Communication overhead**: Increases with processor count
- **Memory bandwidth**: Can become bottleneck

### Weak Scaling (Fixed Work per Processor)

Performance with proportionally increasing problem size:

- **Tree depth**: Grows as log N, maintaining efficiency
- **Load balancing**: Space-filling curves help distribute work
- **Network communication**: Scales well for distributed systems

## Optimization Strategies

### Algorithmic Optimizations

1. **Morton ordering**: Improves cache locality and tree balance
2. **Vectorization**: Process multiple particles simultaneously
3. **Tree reuse**: Rebuild only when necessary for dynamic systems

### Implementation Optimizations

1. **Memory pooling**: Reduce allocation overhead
2. **SIMD instructions**: Vectorize force calculations
3. **Prefetching**: Hide memory latency

### Parallel Optimizations

1. **Shared-memory**: OpenMP for multi-core parallelization
2. **Distributed-memory**: MPI for cluster computing
3. **GPU acceleration**: CUDA/OpenCL for massively parallel systems

## Performance Benchmarks

Typical performance on modern hardware:

| N Particles | Direct O(N²) | Barnes-Hut O(N log N) | Speedup |
|-------------|-------------|----------------------|---------|
| 1,000 | 1ms | 0.1ms | 10× |
| 10,000 | 100ms | 2ms | 50× |
| 100,000 | 10s | 30ms | 333× |
| 1,000,000 | 16min | 400ms | 2,400× |

*Benchmarks assume θ = 0.5 on a modern CPU*