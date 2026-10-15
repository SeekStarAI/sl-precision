// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Quantization and hardware-aware data layout tools for StarLight.
//!
//! This crate is a standalone, tensor-library-independent quantization core.
//! Model readers and writers can convert their weight tensors to `f32`, call
//! [`quantize`], and serialize the returned packed bytes and scales.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::error::Error;
use std::fmt;
use std::ptr::NonNull;
use std::slice;

/// Alignment used for quantized payloads.
pub const PACKED_ALIGNMENT: usize = 64;

/// Integer encoding used for quantized values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationType {
    /// Signed 8-bit values in the range `[-127, 127]`.
    Int8,
    /// Signed 4-bit two's-complement values in the range `[-7, 7]`.
    Int4,
}

impl QuantizationType {
    fn max_code(self) -> f32 {
        match self {
            Self::Int8 => 127.0,
            Self::Int4 => 7.0,
        }
    }

    fn bits(self) -> usize {
        match self {
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }
}

/// Configuration for symmetric, group-wise quantization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizationConfig {
    quantization_type: QuantizationType,
    group_size: usize,
}

impl QuantizationConfig {
    /// Creates a configuration. `group_size` must be greater than zero.
    pub fn new(
        quantization_type: QuantizationType,
        group_size: usize,
    ) -> Result<Self, QuantizationError> {
        if group_size == 0 {
            return Err(QuantizationError::InvalidGroupSize);
        }

        Ok(Self {
            quantization_type,
            group_size,
        })
    }

    /// Returns the selected integer encoding.
    pub fn quantization_type(self) -> QuantizationType {
        self.quantization_type
    }

    /// Returns the number of source values represented by each scale.
    pub fn group_size(self) -> usize {
        self.group_size
    }
}

/// Errors returned by quantization operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizationError {
    /// Quantization groups cannot be empty.
    InvalidGroupSize,
    /// The requested buffer size cannot be represented by the allocator layout.
    BufferTooLarge,
    /// The allocator could not reserve the aligned output buffer.
    AllocationFailed,
    /// Input values must all be finite.
    NonFiniteValue { index: usize },
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGroupSize => formatter.write_str("group size must be greater than zero"),
            Self::BufferTooLarge => formatter.write_str("quantized buffer is too large"),
            Self::AllocationFailed => {
                formatter.write_str("could not allocate aligned quantized buffer")
            }
            Self::NonFiniteValue { index } => {
                write!(formatter, "input value at index {index} is not finite")
            }
        }
    }
}

impl Error for QuantizationError {}

/// A group-wise quantized vector with a 64-byte-aligned payload.
///
/// INT4 values are packed in source order: the even value occupies the low
/// nibble and the odd value occupies the high nibble. The final unused high
/// nibble, if any, is zero. `data()` exposes only the logical payload bytes;
/// allocator padding is not included.
pub struct QuantizedTensor {
    quantization_type: QuantizationType,
    group_size: usize,
    element_count: usize,
    scales: Vec<f32>,
    data: AlignedBuffer,
}

impl QuantizedTensor {
    /// Quantizes a flat slice using the supplied configuration.
    pub fn quantize(values: &[f32], config: QuantizationConfig) -> Result<Self, QuantizationError> {
        quantize(values, config)
    }

    /// Returns the integer encoding used by this tensor.
    pub fn quantization_type(&self) -> QuantizationType {
        self.quantization_type
    }

    /// Returns the number of source elements.
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns the source elements per scale group.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Returns one symmetric scale per group.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Returns the logical packed payload, excluding alignment padding.
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Returns the payload pointer's guaranteed alignment.
    pub fn alignment(&self) -> usize {
        PACKED_ALIGNMENT
    }

    /// Reconstructs the source values into a newly allocated `f32` vector.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.element_count);
        for index in 0..self.element_count {
            let group = index / self.group_size;
            let code = match self.quantization_type {
                QuantizationType::Int8 => self.data.as_slice()[index] as i8 as f32,
                QuantizationType::Int4 => {
                    let packed = self.data.as_slice()[index / 2];
                    let nibble = if index % 2 == 0 {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    let signed = if nibble & 0x08 == 0 {
                        nibble as i8
                    } else {
                        (nibble as i8) - 16
                    };
                    signed as f32
                }
            };
            values.push(code * self.scales[group]);
        }
        values
    }
}

/// Quantizes a flat `f32` slice using symmetric per-group scales.
pub fn quantize(
    values: &[f32],
    config: QuantizationConfig,
) -> Result<QuantizedTensor, QuantizationError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(QuantizationError::NonFiniteValue { index });
    }

    let bit_count = values
        .len()
        .checked_mul(config.quantization_type.bits())
        .ok_or(QuantizationError::BufferTooLarge)?;
    let byte_count = bit_count
        .checked_add(7)
        .ok_or(QuantizationError::BufferTooLarge)?
        / 8;
    let group_count =
        values.len() / config.group_size + usize::from(values.len() % config.group_size != 0);
    let mut data = AlignedBuffer::zeroed(byte_count)?;
    let mut scales = Vec::with_capacity(group_count);
    let data_slice = data.as_mut_slice();
    let max_code = config.quantization_type.max_code();

    for (group_index, group_values) in values.chunks(config.group_size).enumerate() {
        let max_abs = group_values
            .iter()
            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
        let scale = if max_abs == 0.0 {
            1.0
        } else {
            let scale = max_abs / max_code;
            if scale == 0.0 {
                max_abs
            } else {
                scale
            }
        };
        scales.push(scale);

        let start = group_index * config.group_size;
        for (offset, value) in group_values.iter().enumerate() {
            let code = (*value / scale).round().clamp(-max_code, max_code) as i8;
            let index = start + offset;
            match config.quantization_type {
                QuantizationType::Int8 => data_slice[index] = code as u8,
                QuantizationType::Int4 => {
                    let nibble = (code as u8) & 0x0f;
                    if index % 2 == 0 {
                        data_slice[index / 2] = nibble;
                    } else {
                        data_slice[index / 2] |= nibble << 4;
                    }
                }
            }
        }
    }

    Ok(QuantizedTensor {
        quantization_type: config.quantization_type,
        group_size: config.group_size,
        element_count: values.len(),
        scales,
        data,
    })
}

struct AlignedBuffer {
    pointer: NonNull<u8>,
    layout: Layout,
    logical_len: usize,
}

// SAFETY: the allocation is owned by this value and can be moved between threads.
unsafe impl Send for AlignedBuffer {}

// SAFETY: shared access only exposes immutable bytes after quantization completes.
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    fn zeroed(logical_len: usize) -> Result<Self, QuantizationError> {
        let allocation_len = logical_len.max(1);
        let padding = (PACKED_ALIGNMENT - allocation_len % PACKED_ALIGNMENT) % PACKED_ALIGNMENT;
        let allocation_len = allocation_len
            .checked_add(padding)
            .ok_or(QuantizationError::BufferTooLarge)?;
        let layout = Layout::from_size_align(allocation_len, PACKED_ALIGNMENT)
            .map_err(|_| QuantizationError::BufferTooLarge)?;
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or(QuantizationError::AllocationFailed)?;

        Ok(Self {
            pointer,
            layout,
            logical_len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.pointer.as_ptr(), self.logical_len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.pointer.as_ptr(), self.logical_len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::{
        quantize, QuantizationConfig, QuantizationError, QuantizationType, PACKED_ALIGNMENT,
    };

    #[test]
    fn int4_quantizes_by_group_and_packs_signed_nibbles() {
        let config = QuantizationConfig::new(QuantizationType::Int4, 4).unwrap();
        let values = [-7.0, -3.5, 0.0, 7.0, 1.0, -1.0, 0.0];
        let quantized = quantize(&values, config).unwrap();

        assert_eq!(quantized.data()[0], 0xc9);
        assert_eq!(quantized.data()[1], 0x70);
        assert_eq!(quantized.element_count(), values.len());
        assert_eq!(quantized.scales().len(), 2);
        assert_eq!(quantized.data().as_ptr() as usize % PACKED_ALIGNMENT, 0);

        let reconstructed = quantized.dequantize();
        for (index, (original, restored)) in values.iter().zip(reconstructed).enumerate() {
            let scale = quantized.scales()[index / quantized.group_size()];
            assert!((original - restored).abs() <= scale / 2.0 + 1e-6);
        }
        assert_eq!(quantized.data()[3], 0x00);
    }

    #[test]
    fn int8_round_trip_stays_within_half_a_quantization_step() {
        let config = QuantizationConfig::new(QuantizationType::Int8, 3).unwrap();
        let values = [-2.0, 0.25, 1.0, 4.0];
        let quantized = quantize(&values, config).unwrap();
        let reconstructed = quantized.dequantize();

        assert_eq!(quantized.scales().len(), 2);
        assert!(values.iter().zip(reconstructed).enumerate().all(
            |(index, (original, restored))| {
                let scale = quantized.scales()[index / 3];
                (original - restored).abs() <= scale / 2.0 + 1e-6
            }
        ));
    }

    #[test]
    fn empty_and_zero_input_are_supported() {
        let config = QuantizationConfig::new(QuantizationType::Int4, 32).unwrap();
        let empty = quantize(&[], config).unwrap();
        let zeros = quantize(&[0.0; 5], config).unwrap();

        assert!(empty.data().is_empty());
        assert!(empty.scales().is_empty());
        assert_eq!(empty.data().as_ptr() as usize % PACKED_ALIGNMENT, 0);
        assert_eq!(zeros.dequantize(), vec![0.0; 5]);
    }

    #[test]
    fn invalid_configuration_and_non_finite_values_are_rejected() {
        assert_eq!(
            QuantizationConfig::new(QuantizationType::Int4, 0),
            Err(QuantizationError::InvalidGroupSize)
        );
        let config = QuantizationConfig::new(QuantizationType::Int8, 4).unwrap();
        assert_eq!(
            quantize(&[1.0, f32::NAN], config).err(),
            Some(QuantizationError::NonFiniteValue { index: 1 })
        );
    }
}
