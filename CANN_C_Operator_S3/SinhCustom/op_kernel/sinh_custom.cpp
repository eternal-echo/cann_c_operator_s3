#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelArgMaxWithValue {
public:
    __aicore__ inline KernelArgMaxWithValue() {}

    // 初始化
    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputIndiceGM, GM_ADDR outputValuesGM, uint32_t len) {
        // 设置数据总长度
        total_length = len;
        // 设置输入和输出全局内存
        srcGlobal.SetGlobalBuffer((__gm__ T *)(inputGM), total_length);
        dstIndiceGlobal.SetGlobalBuffer((__gm__ int32_t *)(outputIndiceGM), total_length);
        dstValuesGlobal.SetGlobalBuffer((__gm__ T *)(outputValuesGM), total_length);
    }

    // 计算过程
    __aicore__ inline void Process() {
        // 这里我们使用简单的 for 循环来计算
        T maxValue = srcGlobal[0];    // 初始值为第一个元素
        int32_t maxIndex = 0;         // 初始索引为 0

        // 遍历输入数据，找到最大值及其索引
        for (int32_t i = 1; i < total_length; ++i) {
            if (srcGlobal[i] > maxValue) {
                maxValue = srcGlobal[i];
                maxIndex = i;
            }
        }

        // 将计算结果存储到输出数据
        dstValuesGlobal[0] = maxValue;
        dstIndiceGlobal[0] = maxIndex;
    }

private:
    AscendC::GlobalTensor<T> srcGlobal;          // 输入数据
    AscendC::GlobalTensor<T> dstValuesGlobal;    // 输出最大值
    AscendC::GlobalTensor<int32_t> dstIndiceGlobal; // 输出最大值对应的索引

    uint32_t total_length;  // 数据总长度
};

extern "C" __global__ __aicore__ void arg_max_with_value_plugin(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelArgMaxWithValue<half> op;
    //补充init和process函数调用内容
    op.Init(x, y, tiling_data.total_len);
    op.Process();

}
