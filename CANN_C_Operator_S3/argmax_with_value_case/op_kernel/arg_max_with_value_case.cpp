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

template <typename T>
class KernelArgMaxWithValue {
public:
    __aicore__ inline KernelArgMaxWithValue() {}

    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputIndiceGM, GM_ADDR outputValuesGM, GM_ADDR syncGM) {
        uint32_t blockLength = BLOCK_LENGTH;
        if (AscendC::GetBlockIdx() == USE_CORE_NUM - 1) {
            blockLength = TAIL_LENGTH + BLOCK_LENGTH;
        }

        srcGlobal.SetGlobalBuffer((__gm__ T *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        dstIndiceGlobal.SetGlobalBuffer((__gm__ int32_t *)(outputIndiceGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), blockLength);
        dstValuesGlobal.SetGlobalBuffer((__gm__ T *)(outputValuesGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), blockLength);
        syncGlobal.SetGlobalBuffer((__gm__ int32_t *)(syncGM), USE_CORE_NUM * DEFAULT_SYNCALL_NEED_SIZE);

        AscendC::InitGlobalMemory<T>(dstValuesGlobal, blockLength, 0);
        AscendC::InitGlobalMemory<int32_t>(dstIndiceGlobal, blockLength, 0);

        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(T));
        pipe.InitBuffer(workLocalTbuf, BLOCKLEN_CEIL * sizeof(T));
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
        AscendC::LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy(inputLocal[i * BLOCKLEN_CEIL], srcGlobal[i * BLOCK_ELEMENT_NUM], BLOCKLEN_CEIL);
        }
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<T> outputValuesLocal = outQueue.AllocTensor<T>();
        AscendC::LocalTensor<int32_t> outputIndiceLocal = outQueue.AllocTensor<int32_t>();
        AscendC::LocalTensor<T> workLocal = workLocalTbuf.Get<T>();
        AscendC::LocalTensor<T> inputLocal = inQueue.DeQue<T>();

        // Initialize output buffers
        AscendC::Duplicate<T>(outputValuesLocal, 0, BLOCK_GROUP_NUM * BLOCKLEN_CEIL);
        AscendC::Duplicate<int32_t>(outputIndiceLocal, 0, BLOCK_GROUP_NUM * BLOCKLEN_CEIL);
        AscendC::Duplicate<T>(workLocal, 0, BLOCKLEN_CEIL);

        // Mask to control the elements in ReduceMax operation
        uint64_t Mask0 = ((uint64_t)1 << BLOCK_ELEMENT_NUM) - 1;
        uint64_t Mask[2] = {Mask0, 0};

        // ArgMax calculation
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::ReduceMax<T>(outputValuesLocal[i * BLOCKLEN_CEIL], inputLocal[i * BLOCKLEN_CEIL], workLocal, Mask, 1, 8, false);
            AscendC::ArgMax<T>(outputIndiceLocal[i * BLOCKLEN_CEIL], inputLocal[i * BLOCKLEN_CEIL], outputValuesLocal[i * BLOCKLEN_CEIL]);
        }

        outQueue.EnQue<T>(outputValuesLocal);
        outQueue.EnQue<int32_t>(outputIndiceLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<T> outputValuesLocal = outQueue.DeQue<T>();
        AscendC::LocalTensor<int32_t> outputIndiceLocal = outQueue.DeQue<int32_t>();

        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy<T>(dstValuesGlobal[i * BLOCK_ELEMENT_NUM], outputValuesLocal[i * BLOCKLEN_CEIL], BLOCKLEN_CEIL);
            AscendC::DataCopy<int32_t>(dstIndiceGlobal[i * BLOCK_ELEMENT_NUM], outputIndiceLocal[i * BLOCKLEN_CEIL], BLOCKLEN_CEIL);
        }

        outQueue.FreeTensor(outputValuesLocal);
        outQueue.FreeTensor(outputIndiceLocal);
    }

private:
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstValuesGlobal;
    AscendC::GlobalTensor<int32_t> dstIndiceGlobal;
    AscendC::GlobalTensor<int32_t> syncGlobal;
    AscendC::TPipe inQueue;
    AscendC::TPipe outQueue;
    AscendC::TBuf<> workLocalTbuf;
    AscendC::TBuf<> syncLocalTbuf;
};

extern "C" __global__ __aicore__ void arg_max_with_value_case(GM_ADDR x, GM_ADDR indices, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelArgMaxWithValue<DTYPE_X> op;
    op.Init(x, indices, values, workspace);
    op.Process();
}