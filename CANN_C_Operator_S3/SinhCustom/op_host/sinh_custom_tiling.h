
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgMaxWithValueTilingData)
  //考生自行定义tiling结构体成员变量
TILING_DATA_FIELD_DEF(uint32_t, total_len);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgMaxWithValue, ArgMaxWithValueTilingData)
}
