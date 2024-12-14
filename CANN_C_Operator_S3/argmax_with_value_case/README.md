
### 1. 算子功能描述
`ArgMaxWithValue` 算子的功能是计算输入张量沿指定维度的最大值及其对应的索引。这与常见的 `argmax` 操作类似，但额外返回最大值。该算子需要以下几个参数：
- 输入张量：待计算的张量。
- 维度（`dimension`）：指定沿哪个维度计算最大值。
- `keep_dims`：是否保留计算后的维度。

### 2. 算子原型设计
`ArgMaxWithValue` 的原型包括以下几个方面的配置：

- **输入参数**：
  - `x`：输入张量，支持数据类型为 `fp32`, `fp16`, `int32`, `uint8`。
  - `dimension`：指定计算最大值的维度。
  
- **输出参数**：
  - `indices`：返回最大值的索引，数据类型为 `int32`。
  - `values`：返回最大值，数据类型为 `fp32`, `fp16`, `int32`, `uint8`。

- **属性**：
  - `dimension`：指定计算最大值的维度。
  - `keep_dims`：是否保留维度（`bool`，默认值为 `False`）。

### 3. 详细的算子原型 JSON 文件
根据这些设计需求，以下是 `ArgMaxWithValue` 算子的原型 JSON 文件：

```json
[
    {
        "op": "ArgMaxWithValue",  // 算子的类型名称
        "language": "cpp",        // 使用 C++ 编程语言开发
        "input_desc": [           // 输入参数描述
            {
                "name": "x",      // 输入张量的名称
                "param_type": "required",  // 必填
                "format": ["ND"], // 输入格式是多维数组
                "type": ["fp32", "fp16", "int32", "uint8"] // 支持的数据类型
            },
            {
                "name": "dimension",  // 计算最大值的维度
                "param_type": "required",  // 必填
                "format": ["ND"],     // 格式是多维数组（通常为标量或者一维数组）
                "type": ["int32"]     // 维度参数类型是整数
            }
        ],
        "output_desc": [          // 输出参数描述
            {
                "name": "indices",  // 最大值的索引
                "param_type": "required",  // 必填
                "format": ["ND"],   // 输出格式是多维数组
                "type": ["int32"]    // 输出数据类型是整数
            },
            {
                "name": "values",   // 最大值
                "param_type": "required",  // 必填
                "format": ["ND"],   // 输出格式是多维数组
                "type": ["fp32", "fp16", "int32", "uint8"] // 输出数据类型是浮动点数或整数
            }
        ],
        "attributes": [           // 算子的属性描述
            {
                "name": "dimension",  // 必填，指定计算最大值的维度
                "param_type": "required",  // 必填
                "type": "int",        // 属性类型是整数
                "default_value": 0    // 默认值为0，表示计算所有维度
            },
            {
                "name": "keep_dims", // 是否保留维度，默认为 FALSE
                "param_type": "optional",  // 可选
                "type": "bool",       // 布尔类型
                "default_value": false // 默认值为 false
            }
        ]
    }
]
```

### 4. 参数说明

#### 输入参数：
1. **x**：
   - **类型**：`tensor`，支持 `fp32`, `fp16`, `int32`, `uint8` 数据类型。
   - **格式**：`ND`，多维数组。
   - **必填**：是，表示输入张量必须提供。

2. **dimension**：
   - **类型**：`int32`，指定计算最大值的维度。
   - **格式**：`ND`，通常是一维标量（整数），表示沿哪个维度计算最大值。
   - **必填**：是，指定计算最大值的维度。

#### 输出参数：
1. **indices**：
   - **类型**：`tensor`，`int32`，表示最大值的索引。
   - **格式**：`ND`，输出格式为多维数组，与输入的格式一致。
   - **必填**：是，表示输出索引参数必须提供。

2. **values**：
   - **类型**：`tensor`，支持 `fp32`, `fp16`, `int32`, `uint8` 数据类型，表示沿指定维度的最大值。
   - **格式**：`ND`，输出格式为多维数组，与输入的格式一致。
   - **必填**：是，表示输出值参数必须提供。

#### 属性：
1. **dimension**：
   - **类型**：`int`，指定沿哪个维度计算最大值。
   - **必填**：是，必须提供此参数。
   - **默认值**：`0`，表示默认计算第一个维度。

2. **keep_dims**：
   - **类型**：`bool`，指定是否保留维度，默认为 `false`。
   - **必填**：否，属性是可选的。
   - **默认值**：`false`，如果为 `true`，则计算后保持输入张量的维度不变；如果为 `false`，则去掉计算轴。

### 5. 算子计算逻辑
- **ArgMax**：对输入张量 `x`，沿指定的维度 `dimension` 计算最大值的索引，并返回一个新张量 `indices`。返回的是最大值所在的索引位置。
- **Reduce Max**：在 `dimension` 维度上计算最大值，并返回新张量 `values`，它包含该维度上的最大值。

#### 计算过程：
1. **最大值索引**：在 `x` 的指定 `dimension` 维度上，计算该维度的最大值的索引。
2. **最大值计算**：在 `x` 的指定 `dimension` 维度上，计算该维度上的最大值。

### 6. 使用示例
假设输入张量 `x` 为：
```plaintext
x = [
    [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0], [7.1, 8.1, 9.1], [10.0, 11.0, 12.0]]
]
```

如果 `dimension=2`，则：
- **最大值索引**：
  ```plaintext
  indices = [
      [2, 2, 2],  // 每列的最大值索引
      [2, 2, 2]
  ]
  ```
  
- **最大值**：
  ```plaintext
  values = [
      [4.0, 5.0, 6.0],  // 每列的最大值
      [10.0, 11.0, 12.0]
  ]
  ```

### 7. 结论
这个 `ArgMaxWithValue` 算子的设计符合 TensorFlow 和 MindSpore 的规范，支持必要的输入输出参数，以及相关的属性设置。通过 `dimension` 参数来指定计算最大值的维度，通过 `keep_dims` 来控制输出维度是否保持。