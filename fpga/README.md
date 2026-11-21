# FPGA design (AMD Kria KV260)

Placeholder for the hardware side of the project. Planned contents:

- Target device and toolchain (Vivado / Vitis HLS version to be confirmed).
- Ternary or 4-bit datapath design, following the approach in TeLLMe and TerEffic.
- Benchmarks against the published TeLLMe figures (tokens/s, watts, prefill latency) on the same board.

The first design plan for Qwen2.5-0.5B on the KV260 is in [`DESIGN-qwen05b.md`](DESIGN-qwen05b.md): the partition between the CPU and the PL, the datapath, the milestones, and the open questions.

Nothing is implemented yet.
