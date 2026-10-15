# sl-precision

StarLight 的精度与数据排布工具链，负责量化、校准及面向硬件访问模式的权重打包和交织重排。

## 职责边界

- 支持 FP16、BF16、INT8、INT4 等精度表示与转换。
- 输出连续的打包数据及对应缩放、零点等元数据。
- 保持硬件上下文无关，可独立作为库或离线转换工具演进。

接口与实现仍在基线阶段；具体契约以 RFC-0001 为准。

## 开发

在 StarLight workspace 根目录运行 `cargo check -p sl-precision` 或 `cargo test -p sl-precision`。
