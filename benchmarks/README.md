# benchmarks/

Reserved for measurement results.

The point of the project is to reduce texture pop-in under a memory budget, so this directory
will hold the numbers that show whether that worked:

- VRAM / texture-residency measurements under a fixed budget
- pop-in measurements with and without prediction
- prediction accuracy (which mip level was actually needed) and inference latency
- comparison against a naive/prefetch-everything baseline

Nothing here yet - there is no scheduler to benchmark.
