# LAB8: Advanced CUDA Programming & Deep Learning Acceleration

## Overview
This assignment explores advanced GPU computing concepts through hands-on implementation of CUDA kernels and deep learning optimizations. You will learn memory hierarchy utilization, communication patterns, algorithmic efficiency, and practical deep learning acceleration techniques on NVIDIA GPUs.

**GPU: Tesla T4 | CUDA Compute Capability 7.5 | 15GB Memory**

---

## Part 1: Fundamental CUDA Kernels & Bandwidth Analysis

### 1.1 Vector Addition & Performance Analysis

#### Problem: Compute C[i] = A[i] + B[i] for large arrays

```
COMPUTATIONAL PATTERN:
─────────────────────────────────

Element-wise Operation (Embarrassingly Parallel)

CPU:  for i = 0 to N:
        C[i] = A[i] + B[i]
      Serial: O(N)

GPU:  Thread i computes C[i] = A[i] + B[i]
      Parallel: O(1) with N threads
```

### 1.2 Performance Bottlenecks

```
EXECUTION TIMELINE:
───────────────────────────────────

       Memory Transfer (H2D)    GPU Computation    Total Time
           ↓                         ↓               ↓
       ┌────────┐   ┌──────┐   ┌────────┐
       │Transfer│→→→│Compute│→→→│ Result │
       │   ~0.9 ms  │ 0.06ms │   │1.0 ms │
       └────────┘   └──────┘   └────────┘
         Overhead                 Useful Work

Analysis:
- Small N (2^10): ~88% overhead (Transfer dominates)
- Large N (2^26): ~2% overhead (Compute dominates)
- Break-even point: ~2^16 elements
```

### 1.3 Speedup Degradation

```
SPEEDUP vs PROBLEM SIZE:

Speedup (×)
    │
  1 │      ╱─────────── Asymptotic (Memory-Bound Limit)
    │  ╱╱
    │╱  Overhead Dominates
    └────────────────────
      2^10 2^14 2^18 2^22 2^26
      Elements (log scale)

Key Insight:
- Small problems: Negative speedup (overhead > benefit)
- Large problems: Bounded by PCIe bandwidth (~13 GB/s)
```

---

## Part 2A: Launch Configuration Optimization

### Problem: Determine optimal block size for given workload

```
GRID & BLOCK STRUCTURE:
──────────────────────────────────

CUDA Grid:
┌─────────────────────────────────────────┐
│  Block(0,0)    Block(1,0)    Block(2,0) │
│  ┌───────┐    ┌───────┐     ┌───────┐   │
│  │T T T T│    │T T T T│     │T T T T│   │
│  │T T T T│    │T T T T│     │T T T T│   │
│  │T T T T│    │T T T T│     │T T T T│   │
│  │T T T T│    │T T T T│     │T T T T│   │
│  └───────┘    └───────┘     └───────┘   │
└─────────────────────────────────────────┘

Thread Execution Timeline:
Block Size 64:   32 warps executing sequentially
Block Size 256:  8 warps executing in parallel (better latency hiding)
Block Size 512+: Kernel register pressure increases
```

### 1.4 Block Size Impact

```
EXECUTION TIME vs BLOCK SIZE:

Time (ms)
    0.30│     ╲  Block 64
         │      ╲
    0.20│       ╲___  Block 128, 256
         │           ╲ Block 512
    0.10│    ────────╲    ↑ Register Pressure
         │              ╲╱
    0.05│
         └────┬────┬────┬────┬
            64  128 256 512 1024

Optimal: 256 threads/block (balance between occupancy & overhead)
```

---

## Part 2B: Warp Divergence & Control Flow

### Problem: Branch Instructions Impact Performance

```
WARP EXECUTION MODEL (32 threads):
──────────────────────────────────────

Instruction: if (i % 2 == 0) { *2 } else { +1 }

Ideal (No Divergence):
  Thread 0  ✓ (even) ──→ Multiply
  Thread 1  ✗ (odd)  ──→ Multiply  (waiting)
  Thread 2  ✓ (even) ──→ Multiply
  ...
  Thread 31 ✗ (odd)  ──→ Multiply

All 32 threads execute BOTH branches sequentially:
  Pass 1: Execute even-path (odd threads idle)
  Pass 2: Execute odd-path (even threads idle)
  Total: 2× execution time
```

### 1.5 Branch-Free Optimization

```
BRANCHING vs BRANCH-FREE:

Traditional IF-ELSE:
  if (condition) {
    do_work_A();
  } else {
    do_work_B();
  }
  Time: 2T (both branches executed per warp)

Branch-Free (Conditional Assignment):
  result = (condition ? value_A : value_B);
  Time: T (single path, no warp divergence)

Example:
  data[i] = (is_even ? data[i] * 2.0f : data[i] + 1.0f);
  vs.
  if (is_even) data[i] *= 2.0f;
  else data[i] += 1.0f;
```

### Performance Data

```
WARP DIVERGENCE PENALTY:

Scenario              Divergent  Branch-Free  Speedup
─────────────────────────────────────────────────────
Even/Odd split        0.154 ms   0.211 ms    0.73× (interesting!)
(27% slower in divergent case)

This counter-intuitive result suggests:
- GPU compiler optimizations
- Memory access patterns dominate
- Cache efficiency varies
```

---

## Part 3: Reduction Strategies & Memory Optimization

### Problem: Compute global maximum across millions of elements

```
REDUCTION TREE PATTERNS:
────────────────────────────────

Input: [12, 5, 8, 3, 14, 7, 9, 2]

Stride 1:   12-5   8-3   14-7   9-2
            12     8     14     9

Stride 2:   12-8   14-9
            12     14

Stride 4:   12-14
            14  ← Maximum

Complexity: O(log N) rounds, O(N) total operations
```

### 3.1 Shared Memory Tree Reduction

```
SHARED MEMORY APPROACH:
──────────────────────────────────

Per Block:
  1. Load data into shared memory (global → shared)
  2. Synchronize all threads
  3. Each warp reduces locally
  4. Store block result to global memory

Global:
  5. Reduce block results (second kernel pass)

Latency Hiding:
  - Hide global memory latency with shared memory
  - Each thread works on fast SRAM (100GB/s) not global (13GB/s)

Memory Bandwidth Improvement:
  Global Memory: ~13 GB/s (PCIe bottleneck)
  Shared Memory: ~1.7 TB/s (on-chip)
  Factor: 130× faster!
```

### 3.2 Warp Shuffle Reduction

```
SHUFFLE OPERATIONS (No Shared Memory):
─────────────────────────────────────────

Lane 0:  [12] ←→ Lane 16: [7]   result: 12
Lane 1:  [5]  ←→ Lane 17: [9]   result: 9
Lane 2:  [8]  ←→ Lane 18: [2]   result: 8
...

Benefits:
  - No shared memory needed
  - Faster than shared memory approach
  - Register-to-register communication

Performance Ranking:
  1. Warp Shuffle:       190 µs  ← FASTEST
  2. Shared Memory:      258 µs
  3. Sequential (CPU):   433 µs  ← SLOWEST
```

### 3.3 Bank Conflict Analysis

```
SHARED MEMORY BANK ORGANIZATION:
─────────────────────────────────────

32 banks, each 4 bytes

Stride 1:
  Thread 0 → Bank 0  ✓ No conflict
  Thread 1 → Bank 1  ✓
  Thread 2 → Bank 2  ✓
  ...
  Time: 1 cycle

Stride 32:
  Thread 0 → Bank 0  ✗ ALL hit Bank 0!
  Thread 1 → Bank 0  ✗ 32-way conflict
  ...
  Time: 32 cycles (32× SLOWER!)

Stride Timing:
  Stride 1:   27 µs  (no conflict)
  Stride 32:  10 µs  (serialized access, fewer threads)
```

---

## Part 3B: Bank Conflict Solution - Padding

### Problem: Accessing shared memory columns (transpose)

```
UNPADDED 2D TILE (16×16):
──────────────────────────

  tile[row][col]

Vertical access pattern (column-wise):
  Thread 0 reads tile[0][0] → Bank 0
  Thread 1 reads tile[1][0] → Bank 1
  Thread 2 reads tile[2][0] → Bank 2
  ...
  Thread 16 reads tile[16][0] → Bank 0 (conflict with Thread 0!)
  Thread 17 reads tile[17][0] → Bank 1 (conflict with Thread 1!)

Result: 16-way bank conflict
```

```
PADDED 2D TILE (16×17):
────────────────────────

  tile[row][col+1]  ← Extra column shifts banks

Thread 0 reads tile[0][0] → Bank 0
Thread 1 reads tile[1][0] → Bank 1
Thread 2 reads tile[2][0] → Bank 2
...
Thread 16 reads tile[16][0] → Bank 1 (NO conflict with Thread 0!)
Thread 17 reads tile[17][0] → Bank 2 (NO conflict with Thread 1!)

Result: Perfect spacing, no conflicts!
```

### 3B.1 Padding Speedup

```
PERFORMANCE IMPROVEMENT:

                 Time     Speedup
Naive (16×16):  180 µs   1.0×
Padded (16×17): 93  µs   1.93× (Almost 2× faster!)

Cost: Just 1 extra float per row (negligible memory)
Benefit: Massive performance gain
```

---

## Part 3C: Shared Memory Histogram

### Problem: Count frequency of values (0-255) across 2^24 elements

```
HISTOGRAM COMPUTATION:
──────────────────────────────

Global Method (NAIVE):
  for (i = 0 to 2^24):
    atomicAdd(&global_hist[data[i]], 1);
  
  Problem: All threads contend on 256 atomics
  Result: Severe serialization

Shared Memory Method (OPTIMIZED):
  1. Each block allocates private histogram in shared memory
  2. Threads atomicAdd to shared memory (fast, local)
  3. Synchronize block
  4. Merge shared hist into global (fewer contentions)
```

### Efficiency Gain

```
CONTENTION REDUCTION:

Global Atomics:
  All N threads competing for 256 addresses
  Effective bandwidth: Very low (~30%)

Shared Memory Atomics:
  Each block: ~256 threads → 256 addresses
  Low contention within block
  Effective bandwidth: ~80%

Performance:
  Global method:   >100 ms
  Shared method:   0.47 ms  ← 200× faster!
```

---

## Part 4: Machine Learning Kernels

### 4A: Activation Functions

```
ACTIVATION FUNCTIONS (Element-wise):
─────────────────────────────────────

Sigmoid: σ(x) = 1 / (1 + e^(-x))
  ┌─────────────┐
  │      1      │     Range: [0, 1]
  │    ╱╲       │     Smooth, differentiable
  │   ╱  ╲      │     Used: Output layer (binary)
  │  ╱    ╲     │
  │ ╱      ╲    │
  └─────────────┘
  -∞ ← x → +∞

Tanh: tanh(x) = sinh(x) / cosh(x)
  ┌─────────────┐
  │    1  ╱─    │     Range: [-1, 1]
  │    ╱╱       │     Centered at 0
  │   ╱         │     Used: Hidden layers
  │  ───        │
  │ ╱           │
  │╱────  -1    │
  └─────────────┘
  -∞ ← x → +∞

ReLU: f(x) = max(0, x)
  ┌─────────────┐
  │        ╱    │     Range: [0, ∞]
  │      ╱      │     Sparse (50% zeros typically)
  │    ╱        │     Fast to compute
  │  ╱          │     Most popular
  │╱────        │
  │ 0           │
  └─────────────┘
  -∞ ← x → +∞

Leaky ReLU: f(x) = max(αx, x)
  ┌─────────────┐
  │      ╱      │     Range: (-∞, ∞)
  │    ╱        │     Slope for negatives
  │  ╱          │     Prevents "dead neurons"
  │ α╱          │     Trade-off: Complexity
  │╱            │
  │             │
  └─────────────┘
  -∞ ← x → +∞
```

### 4B: Cross-Entropy Loss (Log-Sum-Exp Trick)

```
CROSS-ENTROPY COMPUTATION:
──────────────────────────────

Standard Formula (UNSTABLE):
  loss = -log(e^x_label / Σ e^x_i)
  Problem: e^x overflows for large x!

Log-Sum-Exp Trick (STABLE):
  loss = -(x_label - log(Σ e^(x_i - max_x)) - max_x)
  
  max_x = max(logits)
  log_sum_exp = max_x + log(Σ e^(x_i - max_x))
  loss = -(x_label - log_sum_exp)

Why This Works:
  Subtracting max_x keeps exponents in safe range
  e^(-10) is computable (no overflow)
  e^(10) would overflow double precision
```

### 4C: Adam Optimizer (Fused Kernel)

```
ADAM UPDATE RULE (Per Parameter):
──────────────────────────────────

Input: Parameter p, Gradient g, First moment m, Second moment v

m ← β₁·m + (1-β₁)·g              (Momentum)
v ← β₂·v + (1-β₂)·g²             (Adaptive learning rate)

m̂ ← m / (1 - β₁^t)               (Bias correction)
v̂ ← v / (1 - β₂^t)               (Bias correction)

p ← p - α · m̂ / (√v̂ + ε)        (Update)

Fused Kernel Advantage:
  Traditional: 4 separate kernels (4 global memory round-trips)
  Fused:       1 kernel (1 round-trip)
  Speedup:     ~3-4× for parameter updates
```

---

## Part 5: CNN Layers & Matrix Operations

### 5A: Tiled Matrix Multiplication (GEMM)

```
MATRIX MULTIPLICATION: C = A × B
────────────────────────────────────

Naive: Every thread reads full rows of A and columns of B
  A: 1024×1024, B: 1024×1024
  Global memory: 2×1024 reads per thread
  Latency: ~400 cycles per memory access
  Total: ~800K cycles per thread

Tiled: Break matrices into tiles (16×16)
  Load tile into shared memory (fast, reused 16 times)
  Compute on shared memory data
  Global memory: 1024/16 = 64 phases
  Latency hidden by other tiles

Performance:
  Matrix Size  Naive      Tiled      Speedup
  ──────────────────────────────────────────
  128×128      ~50 GFLOPS  ~98 GFLOPS  2.0×
  256×256      ~100 GFLOPS ~268 GFLOPS 2.7×
  512×512      ~200 GFLOPS ~361 GFLOPS 1.8×
  1024×1024    ~250 GFLOPS ~371 GFLOPS 1.5×
```

### 5B: Max Pooling (2×2)

```
MAX POOLING OPERATION:
──────────────────────────────

Input (4×4):
  ┌─────────────┐
  │ 2  4│ 1  3  │
  │ 5  7│ 2  6  │
  ├─────────────┤
  │ 1  4│ 8  2  │
  │ 3  2│ 9  5  │
  └─────────────┘

Output (2×2):
  ┌──────┐
  │ 7  6 │  (max of each 2×2 block)
  │ 4  9 │
  └──────┘

Spatial Complexity: O(H×W) input → O(H/2×W/2) output
Downsampling: Reduce spatial dimensions without losing features
```

### 5C: Batch Normalization

```
BATCH NORM (Training):
──────────────────────────────────

Input: x (batch of features)

1. Compute batch statistics:
   μ_batch = mean(x)
   σ²_batch = var(x)

2. Normalize:
   x̂ = (x - μ_batch) / √(σ²_batch + ε)

3. Scale and shift:
   y = γ·x̂ + β

Benefits:
  ✓ Stabilizes training (smaller learning rate)
  ✓ Acts as regularizer (reduces overfitting)
  ✓ Allows higher learning rates
  ✓ Less sensitive to weight initialization

Inference:
  Use running mean/variance (computed during training)
  No dependency on batch statistics
```

---

## Part 6: Real-World Application - MNIST CNN Training

### 6A: Model Architecture

```
CNN ARCHITECTURE:
────────────────────────────────

Input: 28×28 MNIST Image

┌─────────────────────────────────────┐
│ Conv(32, 3×3) + ReLU                │ 28×28 → 28×28
│ MaxPool(2×2)                        │ 28×28 → 14×14
│ Conv(64, 3×3) + ReLU                │ 14×14 → 14×14
│ MaxPool(2×2)                        │ 14×14 → 7×7
│ Flatten                             │ 64×7×7 = 3136
│ Dense(128) + ReLU                   │ 3136 → 128
│ Dense(10) + Softmax                 │ 128 → 10 (classes)
└─────────────────────────────────────┘

Output: 10 class probabilities
```

### 6B: Ablation Study Results

```
TRAINING CONFIGURATION COMPARISON:
──────────────────────────────────────

Configuration        Accuracy  Training Time  Speedup
────────────────────────────────────────────────────
1. Baseline (Adam)   98.71%    80.9 s         1.0×
2. + BatchNorm       98.98%    75.3 s         1.1×
3. + Dropout         98.82%    73.9 s         1.1×
4. SGD + CosineLR    98.45%    73.2 s         1.1×

Key Findings:
  ✓ BatchNorm: Slight accuracy boost, faster training
  ✓ Dropout: Prevents overfitting, similar accuracy
  ✓ SGD: Slower convergence than Adam
  ✓ Best: BatchNorm + Adam combination
```

### 6C: Convergence Pattern

```
ACCURACY VS EPOCH:

Accuracy (%)
   100│
      │  ╱─── Baseline (Adam)
    99│ ╱
      │ ╱
    98│─ With BatchNorm
      │ ├── With Dropout
    97│ │
      │ │  SGD + CosineLR
    96│ │  
      │╱╱
    95│───────────────
       │
       └┬──┬──┬──┬──
        1  2  3  4  5
        Epoch

All reach 95% by epoch 1 (good initialization + learning rate)
BatchNorm converges fastest to >98%
```

---

## Key GPU Computing Concepts

### Memory Hierarchy

```
NVIDIA GPU Memory Hierarchy:
────────────────────────────────

Registers:     (~5MB/SM)  ← Fastest (1 cycle latency)
    ↓
Shared Memory: (96KB/SM)  ← Very Fast (20-30 cycles)
    ↓
L1 Cache:      (32KB/SM)  ← Fast (100+ cycles)
    ↓
L2 Cache:      (4MB)      ← Medium (250+ cycles)
    ↓
Global Memory: (15GB)     ← Slow (400+ cycles)
    ↓
PCIe/Host:     (16GB+)    ← Very Slow (1000+ cycles)

Bandwidth:
  Registers:   ~1000 GB/s
  Shared:      ~1700 GB/s
  Global:      ~300 GB/s (T4)
  PCIe:        ~16 GB/s
```

### Occupancy & Latency Hiding

```
GPU EXECUTION MODEL:
────────────────────

Warps (32 threads):  64 warps/SM typically
Active Threads:      ~2000 threads/SM (saturates pipelines)

Latency Hiding:
  When one warp stalls (memory access), GPU switches to another warp
  Result: Memory latency is hidden by computation

Occupancy = Active Warps / Max Warps
  100% occupancy: All pipeline slots filled
  50% occupancy: Half the throughput
```

---

## Performance Optimization Checklist

```
GPU OPTIMIZATION STRATEGY:
──────────────────────────

1. MEMORY ACCESS PATTERNS
   ✓ Coalesced access (consecutive threads read consecutive memory)
   ✓ Avoid bank conflicts in shared memory
   ✓ Use shared memory for frequently accessed data

2. COMPUTATION
   ✓ Minimize warp divergence
   ✓ Use intrinsics (__shfl, __shared__ reduction)
   ✓ Fuse operations (Adam, normalization)

3. BANDWIDTH UTILIZATION
   ✓ Maximize computation-to-memory ratio
   ✓ Hide memory latency with work
   ✓ Overlap computation with transfers

4. OCCUPANCY
   ✓ Keep enough warps active (50-100%)
   ✓ Watch register pressure
   ✓ Balance block size and shared memory usage

5. PROFILING
   ✓ Use NVIDIA Nsight / nvprof
   ✓ Measure memory bandwidth vs peak
   ✓ Identify bottlenecks (compute vs memory)
```

---

## Expected Results Summary

| Part | Concept | Key Metric | Finding |
|------|---------|-----------|---------|
| 1 | Vector Addition | Speedup | Negative for small N, <1× at 2^26 |
| 2A | Launch Config | Block Size | 256 threads optimal for this kernel |
| 2B | Divergence | Time | Branch-free NOT always faster |
| 3A | Reduction | Method | Warp shuffle 2.3× faster than shared |
| 3B | Bank Conflicts | Padding | 1.93× speedup from 16×16 → 16×17 |
| 3C | Histogram | Shared Method | 200× speedup vs global atomics |
| 4 | ML Kernels | Activation | All implemented correctly |
| 5A | GEMM | GFLOPS | ~371 GFLOPS at 1024×1024 |
| 5B | Pooling | Speed | Very fast (<0.02 ms) |
| 6 | MNIST Training | Accuracy | 98.98% with BatchNorm |

---

## Deep Learning Insights

### Why BatchNorm Helps

```
BATCH NORMALIZATION EFFECTS:
───────────────────────────

Training Dynamics:
  ✓ Whitening features (zero mean, unit variance)
  ✓ Reduces internal covariate shift
  ✓ Allows higher learning rates
  ✓ Acts as L2 regularizer

Speed Improvement:
  - Faster convergence (fewer epochs)
  - Better gradient flow
  - Enables faster training

Accuracy:
  - Slight improvement (98.71% → 98.98%)
  - Regularization effect reduces overfitting
  - More stable across different initializations
```

### Dropout vs BatchNorm

```
COMPARISON:
───────────

Dropout:
  - Disables random neurons (50%)
  - Prevents co-adaptation
  - Training: Slower (more stochasticity)
  - Result: 98.82% (good generalization)

BatchNorm:
  - Normalizes features
  - Reduces internal shift
  - Training: Faster
  - Result: 98.98% (slightly better)

Combination (Both):
  - Complements each other
  - Redundant regularization
  - Slightly slower training
```

---

## Advanced Topics & Extensions

### Tensor Cores (Not Used in Assignment)
- Specialized 4×4 matrix multiply units
- Available on A100, RTX series
- ~5-10× speedup for GEMM

### Mixed Precision Training
- FP16 computation + FP32 accumulation
- 2× speedup with minimal accuracy loss
- Requires gradient scaling (overflow prevention)

### Multi-GPU Scaling
- Data parallelism: Distribute batches
- Model parallelism: Distribute model layers
- Requires efficient communication (NCCL)

---

## Key Learning Outcomes

After completing this assignment, you understand:

✓ **GPU Memory Hierarchy**: Registers → Shared → Global trade-offs  
✓ **Warp Execution**: 32-thread SIMT model, synchronization  
✓ **Coalescing & Bank Conflicts**: Memory access optimization  
✓ **Latency Hiding**: Occupancy, instruction parallelism  
✓ **Reduction Patterns**: Tree, shuffle, shared memory  
✓ **Deep Learning Kernels**: Activations, loss, optimization  
✓ **CNN Layers**: GEMM, pooling, batch normalization  
✓ **Real-World Training**: Full pipeline from kernel to application  
✓ **Performance Analysis**: Profiling, bottleneck identification  

---

## Compilation & Execution

### Requirements
```bash
- NVIDIA GPU (Tesla T4 or better)
- CUDA Toolkit 12.8+
- CuPy (Python GPU arrays)
- PyTorch (Deep learning framework)
- NVIDIA cuDNN (CNN optimization library)
```

### Running the Assignment
```bash
# Jupyter Notebook (recommended)
jupyter notebook Assignment8.ipynb

# Part 1-5: Interactive analysis with kernels
# Part 6: MNIST training with ablation study

# Expected output:
# - Performance tables
# - Activation function plots
# - Convergence curves
# - Accuracy comparisons
```

---

## Performance Analysis Template

### Bandwidth Calculation
```
Bandwidth = (Data Transferred in MB) / (Time in seconds)
          = (N elements × 4 bytes per float) / time

Peak Bandwidth (T4): ~321 GB/s
Achieved: ~13-15 GB/s (PCIe dominated)
Efficiency: 4-5%
```

### Speedup Calculation
```
Speedup = T_CPU / T_GPU
Efficiency = Speedup / N_cores
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of Memory | Batch size too large | Reduce batch size or use model parallelism |
| Low GPU Utilization | Kernel too small | Increase problem size or batch size |
| Divergence Penalty | Branch statements | Use conditional operations (ternary operator) |
| Slow Transfers | Host → Device bandwidth | Overlap computation with transfer using streams |
| Numerical Issues | Overflow/underflow | Use log-sum-exp trick, mixed precision |

---

## References
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- NVIDIA GPU Architecture: https://www.nvidia.com/en-us/geforce/graphics-cards/
- CuPy Documentation: https://docs.cupy.dev/
- PyTorch CUDA: https://pytorch.org/
- Optimizing GPU-Accelerated Applications: NVIDIA Best Practices Guide
