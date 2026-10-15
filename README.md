# sl-precision

StarLight 的精度与数据排布工具链，负责量化、校准及面向硬件访问模式的权重打包和交织重排。

它同时是一个独立 Rust crate：不依赖 StarLight 其他模块或张量库，可由模型转换器调用，也可被其他 Rust 项目用作通用量化内核。

## 当前能力

- 对 `f32` 切片执行对称 group-wise INT4 或 INT8 量化。
- INT4 以低 nibble 存偶数索引、高 nibble 存奇数索引，并按二进制补码编码。
- 每组输出一个 scale；量化载荷起始地址按 64 字节对齐。
- 提供反量化 API，用于校验、基准对比和下游工具串接。

当前版本提供 Safetensors CLI 前端。它量化 F32/F16/BF16 张量，其他 dtype 默认原样保留；`--keep` 可重复指定不量化的张量名。输出仍是 Safetensors 容器，但量化权重以 U8 payload 和 F32 scale 张量保存，并通过 `starlight.quantization` 元数据记录恢复映射；它不是可直接交给普通模型加载器的权重文件。

## 使用

```rust
use sl_precision::{quantize, QuantizationConfig, QuantizationType};

let weights = [-1.0_f32, -0.25, 0.25, 1.0];
let config = QuantizationConfig::new(QuantizationType::Int4, 4)?;
let packed = quantize(&weights, config)?;
let payload = packed.data();
let scales = packed.scales();
let restored = packed.dequantize();
```

模型转换示例：

```sh
cargo run --release -p sl-precision --bin sl-quantize -- \
	--input model.safetensors \
	--output model-int4.safetensors \
	--format int4 \
	--group-size 128 \
	--keep model.embed_tokens.weight
```

依赖声明：

```toml
[dependencies]
sl-precision = "0.1"
```

## 设计原则

- 算法不依赖设备、模型框架或张量库，序列化由调用方负责。
- 元数据（scale、元素数、分组大小、格式）与载荷分开，便于适配不同容器。
- 量化阶段允许分配输出缓冲；本 crate 不属于 `sl-core::step` 的运行时热路径。
- 后续扩展须保持编码布局可版本化，并以误差、对齐和格式兼容测试验收。

## 开发

在 StarLight workspace 根目录运行 `cargo test -p sl-precision`。API 约定、量化语义和扩展计划见 [DEVELOPMENT.md](DEVELOPMENT.md)。
