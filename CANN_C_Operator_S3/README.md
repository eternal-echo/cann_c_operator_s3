# 生成工程

使用 [msopgen](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/opdev/optool/atlasopdev_16_0018.html) 工具生成工程。

```bash
cd CANN_C_Operator_S3/argmax_with_value_case
$HOME/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i argmax_with_value_case.json -f tf -c ai_core-ascend910B -lan cpp -out ./
```