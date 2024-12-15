
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

这个代码实现的是一个自定义的 `ReduceMin` 算子，用于计算输入数据的每列最小值。这个算子的特点是采用了无DataCopyPad的非对齐核函数直调方式来处理数据，并且有一定的优化策略，以提高在 Ascend 处理器上的计算效率。下面，我将逐步解析这个算子的原理。

# ReduceMin 算子实现解析

### 1. **算子概述**
`ReduceMin` 算子的功能是计算输入数据沿指定轴的最小值。在这个例子中，`ReduceMin` 的 `numpy` 表达式是：
```python
z = np.min(x, axis=1)
```
即在第1轴（行）上进行最小值的求解。该算子的输入数据是一个形状为 `(16, 4)` 的矩阵，数据类型是 `float16`，格式是 ND 格式（即可以支持任意维度的张量）。

### 2. **代码结构解析**
代码中定义了一个 `KernelReduceMin` 类，这个类负责执行计算任务。它的实现分为几个主要部分：

#### 2.1 **常量定义**
```cpp
constexpr int32_t BLOCK_BYTE_SIZE = 8; // equivalent to the definition of blockLen of DataCopyPad
constexpr int32_t BLOCK_GROUP_NUM = 4; // equivalent to the definition of blockCount of DataCopyPad
constexpr int32_t BLOCK_ELEMENT_NUM = BLOCK_BYTE_SIZE / sizeof(half);
constexpr int32_t BLOCKLEN_CEIL = 32 / sizeof(half); // since BLOCK_BYTE_SIZE<32
constexpr int32_t USE_CORE_NUM = 4;                  // num of core used
constexpr int32_t TILE_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
constexpr int32_t TOTAL_LENGTH = USE_CORE_NUM * TILE_NUM * BUFFER_NUM * BLOCK_GROUP_NUM * BLOCK_ELEMENT_NUM;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TAIL_LENGTH = BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM;    // length of tail block in the last core
```

这些常量定义了数据处理的大小、核心数量、块的大小等。这些设置影响了数据传输和计算的并行性。具体来说：
- `BLOCK_BYTE_SIZE` 定义了每个处理块的字节数。
- `BLOCK_GROUP_NUM` 表示每次处理多少个数据块。
- `BLOCK_ELEMENT_NUM` 是每个块中包含的元素数量。
- `USE_CORE_NUM` 是使用的核心数量。

#### 2.2 **初始化 (`Init`)**
`Init` 函数初始化了计算所需的全局和局部缓冲区，包括：
- 设置源数据（`srcGlobal`）、目标数据（`dstGlobal`）、同步数据（`syncGlobal`）的全局内存缓冲区。
- 清空目标数据区 `dstGlobal`。
- 初始化输入输出队列以及其他工作缓冲区。

```cpp
__aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR syncGM)
{
    uint32_t blockLength = BLOCK_LENGTH;
    if (AscendC::GetBlockIdx() == USE_CORE_NUM - 1) {
        blockLength = TAIL_LENGTH + BLOCK_LENGTH;
    }
    srcGlobal.SetGlobalBuffer((__gm__ half *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    dstGlobal.SetGlobalBuffer((__gm__ half *)(outputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), blockLength);
    syncGlobal.SetGlobalBuffer((__gm__ int32_t *)(syncGM), USE_CORE_NUM * DEFAULT_SYNCALL_NEED_SIZE);
    // clear dstGm before doing calculations
    AscendC::InitGlobalMemory<half>(dstGlobal, blockLength, 0);

    pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
    pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
    pipe.InitBuffer(workLocalTbuf, BLOCKLEN_CEIL * sizeof(half));
    pipe.InitBuffer(syncLocalTbuf, USE_CORE_NUM * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));

    AscendC::LocalTensor<int32_t> SyncLocal = syncLocalTbuf.Get<int32_t>();
    AscendC::SyncAll(syncGlobal, SyncLocal);
}
```

这段代码执行了以下操作：
- 为每个核心设置对应的全局缓冲区。
- 初始化目标数据区，并清空它。
- 设置输入输出队列、局部缓冲区和同步缓冲区。

#### 2.3 **数据复制 (`CopyIn`)**
```cpp
__aicore__ inline void CopyIn(int32_t progress)
{
    AscendC::LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
    for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
        AscendC::DataCopy(inputLocal[i * BLOCKLEN_CEIL], srcGlobal[i * BLOCK_ELEMENT_NUM],
                          BLOCKLEN_CEIL); // each time copy 16 half elements to UB
    }
    inQueue.EnQue(inputLocal);
}
```
`CopyIn` 方法将全局内存中的数据复制到局部缓冲区中。每次复制 `BLOCKLEN_CEIL` 个元素，且每个块的大小为 16 个 `half` 类型元素。

#### 2.4 **计算过程 (`Compute`)**
```cpp
__aicore__ inline void Compute(int32_t progress)
{
    AscendC::LocalTensor<half> outputLocal = outQueue.AllocTensor<half>();
    AscendC::LocalTensor<half> workLocal = workLocalTbuf.Get<half>();
    AscendC::LocalTensor<half> inputLocal = inQueue.DeQue<half>();
    AscendC::Duplicate<half>(outputLocal, 0, BLOCK_GROUP_NUM * BLOCKLEN_CEIL);
    AscendC::Duplicate<half>(workLocal, 0, BLOCKLEN_CEIL);

    uint64_t Mask0 = ((uint64_t)1 << BLOCK_ELEMENT_NUM) - 1; // mask mode controls only the first 4 elements do ReduceMin calculation
    uint64_t Mask[2] = {Mask0, 0};

    // main calculation
    for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
        AscendC::ReduceMin<half>(outputLocal[i * BLOCKLEN_CEIL], inputLocal[i * BLOCKLEN_CEIL], workLocal, Mask, 1,
                                 8, false);
    }
    outQueue.EnQue<half>(outputLocal);
    inQueue.FreeTensor(inputLocal);
}
```
计算过程是该算子的核心。首先，数据会被加载到局部内存，然后通过 `AscendC::ReduceMin` 执行最小值计算。`Mask` 控制哪些元素会参与 `ReduceMin` 计算，这里使用了一个位掩码来控制。

#### 2.5 **数据写回 (`CopyOut`)**
```cpp
__aicore__ inline void CopyOut(int32_t progress)
{
    AscendC::LocalTensor<half> outputLocal = outQueue.DeQue<half>();
    AscendC::SetAtomicAdd<half>();
    for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
        AscendC::DataCopy<half>(dstGlobal[i * BLOCK_ELEMENT_NUM], outputLocal[i * BLOCKLEN_CEIL], BLOCKLEN_CEIL);
    }
    AscendC::SetAtomicNone();
    outQueue.FreeTensor(outputLocal);
}
```
最后，计算结果从局部内存中写回全局内存，确保同步并进行必要的原子操作以避免并发写回冲突。

### 3. **总结**
这个 `ReduceMin` 算子是一个高效的并行计算核函数实现，主要包括：
- **数据并行处理**：通过多个核心并行计算，每个核心处理一部分数据。
- **分块计算**：将数据拆分为多个块（`BLOCK_ELEMENT_NUM`），并在每个块内进行最小值计算。
- **流水线并行**：通过输入队列、输出队列和工作队列，实现了流水线并行，能够在不同的计算阶段同时处理数据。
- **最小值计算**：通过硬件加速的 `ReduceMin` 操作，在局部内存中进行最小值计算，保证了计算效率。

这种设计可以充分利用 Ascend 处理器的并行计算能力，优化了数据的传输和计算过程，适合在大规模数据集上进行高效的最小值计算。

# Arg_MAX

要仿照 `ReduceMin` 算子实现一个 `ArgMaxWithValue` 算子，我们需要根据 `ArgMaxWithValue` 的功能进行设计。`ArgMaxWithValue` 这个算子的功能是返回输入张量中沿指定维度（`dimension`）的最大值的索引和该最大值本身。这个操作的核心是找出最大值的索引和相应的最大值。

### 算子描述
`ArgMaxWithValue` 算子的主要功能是：对于输入的张量 `x`，沿指定的轴（`dimension`）求出每个切片的最大值及其索引。最终输出有两个张量：
1. **索引张量**：`indice`，其数据类型是 `int32`，表示最大值的索引。
2. **值张量**：`values`，其数据类型是与输入相同（可以是 `fp32`, `fp16`, `int32`, `uint8` 等），表示最大值本身。

这个操作和 `tf.argmax` 以及 `tf.reduce_max` 在 TensorFlow 中的功能相似。以下是如何实现 `ArgMaxWithValue` 算子的代码框架。

### 算子规格描述
| 参数 | 名称 | 类型 | 类型范围 | 属性 | 默认值 | 格式 | 参考资料 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- |
| 输入 | x | tensor | fp32, fp16, int32, uint8 | ND | - | - | [tf.argmax](https://tensorflow.google.cn/api_docs/python/tf/math/argmax?hl=en) |
| 输出 | indice | tensor | int32 | ND | - | - | [tf.reduce_max](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_max?hl=en) |
| 输出 | values | tensor | fp32, fp16, int32, uint8 | ND | - | - | |
| 属性 | dimension | int | - | - | - | - | - |
| 属性 | keep_dims | bool | - | FALSE | - | - | - |

### 1. 核心操作分析

在 `ArgMaxWithValue` 操作中，我们的目标是沿某个维度（`dimension`）对输入进行操作，分别计算每个切片的最大值及其索引。与 `ReduceMin` 相比，这里需要同时返回两个输出：一个是最大值的索引（`indices`），另一个是最大值本身（`values`）。因此，算子的计算将分为两步：
1. **计算最大值**：利用硬件加速的 `ReduceMax` 函数来计算每个切片的最大值。
2. **计算索引**：根据最大值所在的位置返回其索引。

### 2. 算子设计

我们可以仿照 `ReduceMin` 算子的结构来设计 `ArgMaxWithValue` 算子，主要涉及以下几个步骤：

#### 2.1 **初始化部分**

首先，需要为 `ArgMaxWithValue` 算子准备全局缓冲区、局部缓冲区以及同步缓冲区，初始化输入输出队列。

#### 2.2 **数据复制**

将输入数据从全局内存复制到局部内存，以便进行并行计算。在 `ArgMaxWithValue` 中，每个核心会计算对应维度上的最大值及其索引。

#### 2.3 **计算最大值和索引**

对于每个数据块，我们需要：
- 计算该块的最大值。
- 根据最大值的位置计算索引。

这里我们需要两个输出：
1. `indices`：表示最大值索引。
2. `values`：表示最大值。

#### 2.4 **写回输出**

计算完成后，将 `indices` 和 `values` 写回到全局内存。

### 3. 代码实现

```cpp
#include "kernel_operator.h"
constexpr int32_t BLOCK_BYTE_SIZE = 8;
constexpr int32_t BLOCK_GROUP_NUM = 4;
constexpr int32_t BLOCK_ELEMENT_NUM = BLOCK_BYTE_SIZE / sizeof(half);
constexpr int32_t BLOCKLEN_CEIL = 32 / sizeof(half);
constexpr int32_t USE_CORE_NUM = 4;
constexpr int32_t TILE_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
constexpr int32_t TOTAL_LENGTH = USE_CORE_NUM * TILE_NUM * BUFFER_NUM * BLOCK_GROUP_NUM * BLOCK_ELEMENT_NUM;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TAIL_LENGTH = BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM;

class KernelArgMaxWithValue {
public:
    __aicore__ inline KernelArgMaxWithValue() {}

    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputIndiceGM, GM_ADDR outputValuesGM, GM_ADDR syncGM) {
        uint32_t blockLength = BLOCK_LENGTH;
        if (AscendC::GetBlockIdx() == USE_CORE_NUM - 1) {
            blockLength = TAIL_LENGTH + BLOCK_LENGTH;
        }

        srcGlobal.SetGlobalBuffer((__gm__ half *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        dstIndiceGlobal.SetGlobalBuffer((__gm__ int32_t *)(outputIndiceGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), blockLength);
        dstValuesGlobal.SetGlobalBuffer((__gm__ half *)(outputValuesGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), blockLength);
        syncGlobal.SetGlobalBuffer((__gm__ int32_t *)(syncGM), USE_CORE_NUM * DEFAULT_SYNCALL_NEED_SIZE);

        AscendC::InitGlobalMemory<half>(dstValuesGlobal, blockLength, 0);
        AscendC::InitGlobalMemory<int32_t>(dstIndiceGlobal, blockLength, 0);

        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(workLocalTbuf, BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(syncLocalTbuf, USE_CORE_NUM * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));

        AscendC::LocalTensor<int32_t> SyncLocal = syncLocalTbuf.Get<int32_t>();
        AscendC::SyncAll(syncGlobal, SyncLocal);
    }

    __aicore__ inline void Process() {
        const int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy(inputLocal[i * BLOCKLEN_CEIL], srcGlobal[i * BLOCK_ELEMENT_NUM], BLOCKLEN_CEIL);
        }
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<half> outputValuesLocal = outQueue.AllocTensor<half>();
        AscendC::LocalTensor<int32_t> outputIndiceLocal = outQueue.AllocTensor<int32_t>();
        AscendC::LocalTensor<half> workLocal = workLocalTbuf.Get<half>();
        AscendC::LocalTensor<half> inputLocal = inQueue.DeQue<half>();
        
        AscendC::Duplicate<half>(outputValuesLocal, 0, BLOCK_GROUP_NUM * BLOCKLEN_CEIL);
        AscendC::Duplicate<int32_t>(outputIndiceLocal, 0, BLOCK_GROUP_NUM * BLOCKLEN_CEIL);
        AscendC::Duplicate<half>(workLocal, 0, BLOCKLEN_CEIL);

        uint64_t Mask0 = ((uint64_t)1 << BLOCK_ELEMENT_NUM) - 1;
        uint64_t Mask[2] = {Mask0, 0};

        // ArgMax calculation
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::ReduceMax<half>(outputValuesLocal[i * BLOCKLEN_CEIL], inputLocal[i * BLOCKLEN_CEIL], workLocal, Mask, 1, 8, false);
            AscendC::ArgMax<half>(outputIndiceLocal[i * BLOCKLEN_CEIL], inputLocal[i * BLOCKLEN_CEIL], outputValuesLocal[i * BLOCKLEN_CEIL]);
        }

        outQueue.EnQue<half>(outputValuesLocal);
        outQueue.EnQue<int32_t>(outputIndiceLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<half> outputValuesLocal = outQueue.DeQue<half>();
        AscendC::LocalTensor<int32_t> outputIndiceLocal = outQueue.DeQue<int32_t>();

        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy<half>(dstValuesGlobal[i * BLOCK_ELEMENT_NUM], outputValuesLocal[i * BLOCK

LEN_CEIL], BLOCKLEN_CEIL);
            AscendC::DataCopy<int32_t>(dstIndiceGlobal[i * BLOCK_ELEMENT_NUM], outputIndiceLocal[i * BLOCKLEN_CEIL], BLOCKLEN_CEIL);
        }

        outQueue.FreeTensor(outputValuesLocal);
        outQueue.FreeTensor(outputIndiceLocal);
    }

private:
    AscendC::Buffer<half> srcGlobal;
    AscendC::Buffer<half> dstValuesGlobal;
    AscendC::Buffer<int32_t> dstIndiceGlobal;
    AscendC::Buffer<int32_t> syncGlobal;
    AscendC::Pipe inQueue;
    AscendC::Pipe outQueue;
    AscendC::Buffer<half> workLocalTbuf;
    AscendC::Buffer<int32_t> syncLocalTbuf;
};
```

### 4. **代码讲解**

1. **初始化 (`Init`)**:
   - 设置输入输出缓冲区，将输入数据加载到全局内存，输出结果（`indices` 和 `values`）写回全局内存。

2. **数据复制 (`CopyIn`)**:
   - 将输入数据从全局内存复制到局部内存，便于并行计算。

3. **计算部分 (`Compute`)**:
   - 在每个核心上，首先计算每个数据块的最大值 (`ReduceMax`)，然后通过 `ArgMax` 查找最大值的索引。

4. **输出 (`CopyOut`)**:
   - 将计算结果写回全局内存，包括最大值 (`values`) 和最大值索引 (`indices`)。

### 5. **总结**

这个 `ArgMaxWithValue` 算子的实现，基于 `ReduceMin` 算子的框架，适当调整了计算逻辑，增加了对最大值的计算和索引提取。通过精心的内存管理和并行计算，我们可以高效地实现该算子，并充分利用硬件加速的计算能力。

# 参考资料
## 规约操作 [ReduceMax](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/apiref/opdevgapi/atlasascendc_api_07_0084.html)接口

`ReduceMax` 是一个用于在输入数据中找到最大值及其索引位置的函数。在硬件加速平台上，尤其是在处理大规模数据时，它常用于减少操作，特别是在神经网络中进行降维、池化等操作时。我们通过几个具体的例子来讲解它的功能、输入输出及其使用方式。

### 1. 功能概述

`ReduceMax` 函数的作用是找出输入数据中的最大值及其对应的索引位置，并将结果存储到指定的目标 `dstLocal` 中。该函数接受多个输入参数，允许用户指定是否要返回最大值的索引。根据 `calIndex` 参数的设置，输出结果可能仅包含最大值，或者同时包含最大值及其索引。

### 2. 参数解析

- **`dstLocal`**: 输出张量，存储结果。类型为 `LocalTensor`。
- **`srcLocal`**: 输入张量，包含待处理的数据。类型也为 `LocalTensor`。
- **`workLocal`**: 中间结果存储区域，必须根据计算量的大小来分配合适的空间。
- **`count`**: 输入数据元素的数量，用于前 N 个数据的计算。
- **`calIndex`**: 一个布尔值，决定是否计算并返回最大值的索引。`true` 时返回最大值和索引，`false` 只返回最大值。
- **`mask`**: 控制哪些数据参与计算的掩码。可采用逐 bit 模式或者连续模式来控制。
- **`repeatTimes`**: 重复迭代次数，用于处理数据块的多次迭代。
- **`srcRepStride`**: 用于高维切分计算时，源数据块地址的步长。

### 3. 示例讲解

#### 3.1. 示例一：前 N 个数据计算

假设我们有 8320 个数据，我们希望找出最大值及其索引，并且不对数据进行切分。

```cpp
ReduceMax<half>(dstLocal, srcLocal, workLocal, 8320, true);
```

在这个例子中：

- **`srcLocal`**: 包含 8320 个数据的输入张量。
- **`dstLocal`**: 输出张量，存储最大值及其索引。
- **`workLocal`**: 中间数据区域，大小需要根据计算的复杂度来估算。
- **`count = 8320`**: 输入数据的数量。
- **`calIndex = true`**: 同时返回最大值及其索引。

##### 计算步骤

1. **寻找最大值**: 执行一次 `ReduceMax` 操作，找到 `srcLocal` 中的最大值。
2. **计算索引**: 在找到最大值的同时，计算出该最大值在 `srcLocal` 中的位置索引。

##### 输出结果

假设最大值为 `0.9985`，其索引为 `114`，那么：

- `dstLocal` 中的第一个元素为最大值 `0.9985`。
- 第二个元素为最大值的索引 `114`（需要转换成整数格式）。

#### 3.2. 示例二：高维切分计算（逐 bit 模式）

在这种模式下，我们可以按 bit 控制每次迭代计算哪些元素。例如，我们要处理 8320 个数据，使用逐 bit 模式来选择哪些数据参与计算。

```cpp
uint64_t mask[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
ReduceMax<half>(dstLocal, srcLocal, workLocal, mask, 65, 8, true);
```

在此示例中：

- **`mask[2] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}`**: 选择所有数据参与计算。每个 `1` 表示对应的元素会参与计算。
- **`repeatTimes = 65`**: 重复迭代 65 次，确保每次读取 256 字节数据。
- **`srcRepStride = 8`**: 数据块步长为 8，表示每次迭代跳过的块数。

##### 计算步骤

1. **数据划分**: 将 8320 个数据按块划分，并使用 `mask` 来控制每次迭代中哪些元素参与计算。
2. **迭代计算**: 每次迭代计算一部分数据的最大值及索引，直到处理完所有数据。

##### 输出结果

最终，`dstLocal` 将包含最大值和最大值索引。例如，假设最大值为 `0.983`，索引为 `502`，那么：

- `dstLocal[0] = 0.983`
- `dstLocal[1] = 502`

#### 3.3. 示例三：使用连续模式的高维切分计算

在连续模式下，我们指定每次迭代参与计算的数据块大小。

```cpp
uint64_t mask = 128;
ReduceMax<half>(dstLocal, srcLocal, workLocal, mask, 65, 8, true);
```

在此例中，`mask = 128` 表示每次迭代选择 128 个连续的数据进行计算。

##### 计算步骤

1. **数据划分**: 将数据划分为多个连续的数据块，每个数据块大小由 `mask` 控制。
2. **迭代计算**: 每次迭代计算一个数据块的最大值及索引，直到处理完所有数据。

##### 输出结果

假设最终的最大值为 `0.95`，索引为 `320`，输出结果为：

- `dstLocal[0] = 0.95`
- `dstLocal[1] = 320`

### 4. 总结

`ReduceMax` 是一个高效的操作，它可以用于处理大量数据并找出最大值及其索引。通过不同的模式（如逐 bit 模式和连续模式），我们可以灵活地控制每次计算的数据范围。这对于大规模数据处理、深度学习中的池化操作等场景非常有用。在实际应用中，合适地选择 `calIndex` 和 `mask` 参数能够显著提高计算效率。