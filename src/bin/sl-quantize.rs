// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

use half::{bf16, f16};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use safetensors::SafeTensors;
use serde_json::json;
use sl_precision::{quantize, QuantizationConfig, QuantizationType};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const QUANTIZATION_METADATA_KEY: &str = "starlight.quantization";

struct Arguments {
    input: PathBuf,
    output: PathBuf,
    config: QuantizationConfig,
    keep: HashSet<String>,
}

struct OutputTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sl-quantize: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(arguments) = parse_arguments()? else {
        return Ok(());
    };
    let quantized_count = convert_model(
        &arguments.input,
        &arguments.output,
        arguments.config,
        &arguments.keep,
    )?;

    println!(
        "quantized {quantized_count} tensors to {}",
        arguments.output.display()
    );
    Ok(())
}

fn parse_arguments() -> Result<Option<Arguments>, Box<dyn Error>> {
    let values: Vec<String> = std::env::args().skip(1).collect();
    if values.is_empty()
        || values
            .iter()
            .any(|value| value == "--help" || value == "-h")
    {
        print_help();
        return Ok(None);
    }

    let mut input = None;
    let mut output = None;
    let mut format = QuantizationType::Int4;
    let mut group_size = 128;
    let mut keep = HashSet::new();
    let mut index = 0;

    while index < values.len() {
        let option = values[index].as_str();
        index += 1;
        let value = values.get(index).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("missing value for {option}"),
            )
        })?;
        index += 1;

        match option {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--group-size" => group_size = value.parse()?,
            "--format" => {
                format = match value.as_str() {
                    "int4" => QuantizationType::Int4,
                    "int8" => QuantizationType::Int8,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "format must be int4 or int8",
                        )
                        .into())
                    }
                }
            }
            "--keep" => {
                keep.insert(value.clone());
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown option: {option}"),
                )
                .into())
            }
        }
    }

    let input =
        input.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--input is required"))?;
    let output = output
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--output is required"))?;
    let config = QuantizationConfig::new(format, group_size)?;

    Ok(Some(Arguments {
        input,
        output,
        config,
        keep,
    }))
}

fn convert_model(
    input_path: &Path,
    output_path: &Path,
    config: QuantizationConfig,
    keep: &HashSet<String>,
) -> Result<usize, Box<dyn Error>> {
    let input_path = fs::canonicalize(input_path)?;
    if output_path.exists() && fs::canonicalize(output_path)? == input_path {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "input and output paths must be different",
        )
        .into());
    }
    if output_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("output already exists: {}", output_path.display()),
        )
        .into());
    }

    let input_bytes = fs::read(&input_path)?;
    let (_, source_metadata) = SafeTensors::read_metadata(&input_bytes)?;
    let input_file = SafeTensors::deserialize(&input_bytes)?;
    let mut output_tensors = Vec::new();
    let mut output_names = HashSet::new();
    let mut input_names = HashSet::new();
    let mut tensor_manifest = BTreeMap::new();
    let mut quantized_count = 0;

    for (name, tensor) in input_file.tensors() {
        input_names.insert(name.clone());
        if keep.contains(&name) {
            push_tensor(
                &mut output_tensors,
                &mut output_names,
                OutputTensor {
                    name,
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    data: tensor.data().to_vec(),
                },
            )?;
            continue;
        }

        let Some(values) = decode_float_tensor(&tensor)? else {
            push_tensor(
                &mut output_tensors,
                &mut output_names,
                OutputTensor {
                    name,
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    data: tensor.data().to_vec(),
                },
            )?;
            continue;
        };

        let quantized = quantize(&values, config)?;
        let payload_name = format!("{name}.__sl_quantized_data");
        let scales_name = format!("{name}.__sl_scales");
        let original_dtype = tensor.dtype().to_string();
        let original_shape = tensor.shape().to_vec();
        let scales_shape = vec![quantized.scales().len()];
        let scale_bytes: Vec<u8> = quantized
            .scales()
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .collect();

        push_tensor(
            &mut output_tensors,
            &mut output_names,
            OutputTensor {
                name: payload_name.clone(),
                dtype: Dtype::U8,
                shape: vec![quantized.data().len()],
                data: quantized.data().to_vec(),
            },
        )?;
        push_tensor(
            &mut output_tensors,
            &mut output_names,
            OutputTensor {
                name: scales_name.clone(),
                dtype: Dtype::F32,
                shape: scales_shape,
                data: scale_bytes,
            },
        )?;

        tensor_manifest.insert(
            name,
            json!({
                "payload": payload_name,
                "scales": scales_name,
                "shape": original_shape,
                "source_dtype": original_dtype,
            }),
        );
        quantized_count += 1;
    }

    if let Some(missing_name) = keep.difference(&input_names).next() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("--keep tensor does not exist: {missing_name}"),
        )
        .into());
    }
    if quantized_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no supported floating-point tensors were selected for quantization",
        )
        .into());
    }

    let mut metadata = source_metadata.metadata().clone().unwrap_or_default();
    metadata.insert(
        QUANTIZATION_METADATA_KEY.to_string(),
        json!({
            "version": 1,
            "format": match config.quantization_type() {
                QuantizationType::Int4 => "int4",
                QuantizationType::Int8 => "int8",
            },
            "group_size": config.group_size(),
            "layout": "even-index-low-nibble",
            "scales_dtype": "F32",
            "tensors": tensor_manifest,
        })
        .to_string(),
    );

    let views: Vec<(&str, TensorView<'_>)> = output_tensors
        .iter()
        .map(|tensor| {
            TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)
                .map(|view| (tensor.name.as_str(), view))
        })
        .collect::<Result<_, _>>()?;
    serialize_to_file(views, Some(metadata), output_path)?;
    Ok(quantized_count)
}

fn decode_float_tensor(tensor: &TensorView<'_>) -> Result<Option<Vec<f32>>, io::Error> {
    let values = match tensor.dtype() {
        Dtype::F32 => tensor
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect(),
        Dtype::F16 => tensor
            .data()
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect(),
        Dtype::BF16 => tensor
            .data()
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect(),
        _ => return Ok(None),
    };
    Ok(Some(values))
}

fn push_tensor(
    tensors: &mut Vec<OutputTensor>,
    names: &mut HashSet<String>,
    tensor: OutputTensor,
) -> Result<(), io::Error> {
    if !names.insert(tensor.name.clone()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "generated tensor name collides with an input tensor: {}",
                tensor.name
            ),
        ));
    }
    tensors.push(tensor);
    Ok(())
}

fn print_help() {
    println!(
        "Usage: sl-quantize --input MODEL.safetensors --output OUTPUT.safetensors [options]\n\
         Options:\n\
         --format int4|int8       Quantization format (default: int4)\n\
         --group-size N           Values per scale group (default: 128)\n\
         --keep TENSOR_NAME       Preserve a tensor; may be repeated\n\
         -h, --help               Show this help"
    );
}

#[cfg(test)]
mod tests {
    use super::convert_model;
    use half::{bf16, f16};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use safetensors::SafeTensors;
    use sl_precision::{QuantizationConfig, QuantizationType};
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn converts_float_tensors_and_preserves_other_tensors_and_metadata() {
        let input_path = temporary_path("input");
        let output_path = temporary_path("output");
        let weight_bytes: Vec<u8> = [-2.0_f32, -1.0, 0.0, 1.0, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let f16_bytes: Vec<u8> = [f16::from_f32(-1.0), f16::from_f32(1.0)]
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let bf16_bytes: Vec<u8> = [bf16::from_f32(-2.0), bf16::from_f32(2.0)]
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let token_bytes = [3_i32, 7_i32]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let input_tensors = vec![
            (
                "layer.weight",
                TensorView::new(Dtype::F32, vec![5], &weight_bytes).unwrap(),
            ),
            (
                "layer.f16_weight",
                TensorView::new(Dtype::F16, vec![2], &f16_bytes).unwrap(),
            ),
            (
                "layer.bf16_weight",
                TensorView::new(Dtype::BF16, vec![2], &bf16_bytes).unwrap(),
            ),
            (
                "token_ids",
                TensorView::new(Dtype::I32, vec![2], &token_bytes).unwrap(),
            ),
        ];
        let source_metadata = HashMap::from([("model_type".to_string(), "test".to_string())]);
        serialize_to_file(
            input_tensors
                .iter()
                .map(|(name, tensor)| ((*name).to_string(), tensor)),
            Some(source_metadata),
            &input_path,
        )
        .unwrap();

        let config = QuantizationConfig::new(QuantizationType::Int4, 4).unwrap();
        let count = convert_model(&input_path, &output_path, config, &HashSet::new()).unwrap();
        let output_bytes = fs::read(&output_path).unwrap();
        let output_file = SafeTensors::deserialize(&output_bytes).unwrap();

        assert_eq!(count, 3);
        assert_eq!(
            output_file
                .tensor("layer.weight.__sl_quantized_data")
                .unwrap()
                .dtype(),
            Dtype::U8
        );
        assert_eq!(
            output_file
                .tensor("layer.weight.__sl_scales")
                .unwrap()
                .dtype(),
            Dtype::F32
        );
        assert_eq!(
            output_file
                .tensor("layer.f16_weight.__sl_quantized_data")
                .unwrap()
                .dtype(),
            Dtype::U8
        );
        assert_eq!(
            output_file
                .tensor("layer.bf16_weight.__sl_quantized_data")
                .unwrap()
                .dtype(),
            Dtype::U8
        );
        assert_eq!(output_file.tensor("token_ids").unwrap().data(), token_bytes);
        let (_, output_metadata) = SafeTensors::read_metadata(&output_bytes).unwrap();
        let metadata = output_metadata.metadata().as_ref().unwrap();
        assert_eq!(metadata.get("model_type").unwrap(), "test");
        assert!(metadata.contains_key("starlight.quantization"));

        fs::remove_file(input_path).unwrap();
        fs::remove_file(output_path).unwrap();
    }

    fn temporary_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sl-precision-{name}-{}.safetensors",
            std::process::id()
        ))
    }
}
