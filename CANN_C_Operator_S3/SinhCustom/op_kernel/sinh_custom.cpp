#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename DTYPE>
class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t total_size, uint32_t tile_num)
    {
        //考生补充初始化代码
        assert(GetBlockNum() != 0, "block num is 0");
        this->block_len = total_size / GetBlockNum();
        this->tile_num = tile_num;

        assert(tile_num != 0, "tile num can not be zero!");
        this->tile_len = this->block_len / tile_num;

        xGm.SetGlobalBuffer((__gm__ DTYPE*)x + this->block_len * GetBlockIdx(), this->block_len);
        yGm.SetGlobalBuffer((__gm__ DTYPE*)y + this->block_len * GetBlockIdx(), this->block_len);
        pipe.InitBuffer(inQueueX,BUFFER_NUM, this->tile_len * sizeof(DTYPE));
        pipe.InitBuffer(outQueueY,BUFFER_NUM, this->tile_len * sizeof(DTYPE));
    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->tile_num * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tile_len], this->tile_len);
        inQueueX.EnQue<DTYPE>(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> yLocal = outQueueY.AllocTensor<DTYPE>();

        // \text{sinh}(x) = \frac{e^x - e^{-x}}{2}
        AscendC::Exp(xLocal, xLocal, this->tile_len); // e^{x}, 0
        AscendC::Reciprocal(yLocal, xLocal, this->tile_len); // e^{x}, e^{-x}
        AscendC::Sub(yLocal, xLocal, yLocal, this->tile_len); // e^{x}, e^{x} - e^{-x}
        AscendC::Muls(yLocal, yLocal, (half)0.5, this->tile_len); // e^{x}, \frac{e^x - e^{-x}}{2}
        outQueueY.EnQue<DTYPE>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE> yLocal = outQueueY.DeQue<DTYPE>();
        AscendC::DataCopy(yGm[progress * this->tile_len], yLocal, this->tile_len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量
    uint32_t block_len;
    uint32_t tile_num;
    uint32_t tile_len;
};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinh<half> op;
    //补充init和process函数调用内容
    op.Init(x, y, tiling_data.total_len, tiling_data.tile_num);
    op.Process();

}
