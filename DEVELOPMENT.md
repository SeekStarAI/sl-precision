# sl-precision Development Guide

## Scope

`sl-precision` owns hardware-independent numeric conversion and packed weight layouts. It must remain usable without `sl-core`, a tensor framework, or a device context. Model-format adapters and command-line frontends belong above this crate and should call its public API.

The crate may allocate while converting or preparing weights. Zero-allocation requirements apply to inference execution paths in `sl-core` and backends, not to offline quantization. The `sl-quantize` binary is a thin Safetensors frontend over the same public quantization API.

## Current Quantization Contract

- Input: a flat slice of finite `f32` values.
- Supported formats: symmetric signed INT4 and INT8.
- Scales: one `f32` scale per consecutive group; zero-only groups use scale `1.0`.
- INT4 range: `[-7, 7]`; signed values use four-bit two's-complement encoding. `-8` is intentionally unused so positive and negative magnitudes have the same limit.
- INT8 range: `[-127, 127]`; `-128` is intentionally unused for the same symmetric range.
- Rounding: nearest integer using Rust `f32::round`, then clamp to the format's symmetric range.
- INT4 ordering: even element index in the low nibble, odd element index in the high nibble. A final unused high nibble is zero.
- Payload: logical byte length is `ceil(elements * bits / 8)`. The payload pointer is 64-byte aligned; allocator padding is not exposed by `data()` and must not be serialized.
- Scales are stored separately from the packed payload and indexed by `element_index / group_size`.
- NaN and infinite inputs are rejected with the offending element index.

## Public API

`QuantizationConfig::new` validates the group size. `quantize` returns a `QuantizedTensor` exposing its format, element count, group size, scales, logical payload and alignment. `dequantize` reconstructs values into a newly allocated `Vec<f32>`; it is intended for conversion workflows and tests, not inference hot paths.

The payload is a primitive buffer, not a complete model artifact. `sl-quantize` replaces each selected floating tensor with `<name>.__sl_quantized_data` (U8) and `<name>.__sl_scales` (F32); original tensor names, shapes, source dtypes and grouping information are recorded in the `starlight.quantization` Safetensors metadata entry. Existing Safetensors metadata is retained. This output requires a StarLight-aware loader and is not interchangeable with ordinary F16/BF16 model files.

## Validation

Run from the workspace root:

```sh
cargo test -p sl-precision
cargo test -p sl-precision --bin sl-quantize
cargo clippy -p sl-precision --all-targets -- -D warnings
```

Run the model converter with `cargo run -p sl-precision --bin sl-quantize -- --input model.safetensors --output model-int4.safetensors --format int4 --group-size 128`. The current frontend reads the input file into memory; memory-mapped streaming is a future improvement for large checkpoints.

Tests should cover quantization error bounds per group, exact nibble ordering, odd element counts, empty tensors, zero-only groups, non-finite rejection, and the 64-byte payload alignment guarantee.

## Planned Work

1. Define stable serialized metadata and a versioned tensor representation.
2. Add memory-mapped/streaming conversion for large checkpoints.
3. Add tensor selection policies beyond exact-name `--keep` and validation against representative models.
4. Add hardware-specific interleaving as opt-in layouts, keeping the canonical packed format stable.
5. Add calibration methods only with dedicated numeric tests and documented error semantics.