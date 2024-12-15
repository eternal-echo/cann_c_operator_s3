
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SinhCustomTilingData)
  //考生自行定义tiling结构体成员变量
TILING_DATA_FIELD_DEF(uint32_t, total_len);
TILING_DATA_FIELD_DEF(uint32_t, tile_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SinhCustom, SinhCustomTilingData)
}
