# EyeReal对接助教接口

本目录从助教`testing/inference.py`复制接口后实现`create_predictor`，原始`testing/`与`data_collection/`保持不变。用于实际运行的模型、求解器和参数见`../lightfield_lab/exam_adapter.py`、`exam_geometry.py`。

## 本机直接运行

在`development/`目录执行：

```powershell
& 'D:\anaconda\envs\test\python.exe' .\solution\inference.py --index .\testing\demo_input\test.json --out ..\group_new --group group_new --steps 30 --device cuda
```

`--out`必须是新的空目录。默认从本项目`weights/pretrained_model.pth`加载权重，也可以用`--weights`指定。支持`--steps`、`--lr`、`--phase-steps`、`--anchor-weight`、`--time-limit`、`--reference-score`，默认30步、学习率0.02、无相位暖启动和图样正则、整套600秒软预算、GPU自测。

正常模式实际计算图样，不使用占位图。`--format-demo`与`--submission`仍保留助教接口原有语义。入口同时生成：

```text
group_001/
  REPORT.md
  config.json
  metrics.json
  delivery.json
  code/
    inference.py
    inference.sh
    create_env.sh
    check_submission.py
    ta_reference.py
    requirements.txt
    LICENSE
    THIRD_PARTY_NOTICES.md
    lightfield_lab/*.py
    EyeReal/model/*.py
    EyeReal/data/*.py
    EyeReal/config/*.py
    EyeReal/LICENSE
  weights/pretrained_model.pth
  patterns.json
  patterns/*.png
  eye_images.json
  eye_images/*.png
  masks/*.png
  demo/index.html
```

导出目录包含实际推理所用的自定义模块和权重，复现不依赖原工作区。外部输入清单和目标图按考试发放，不放进模型权重。自动生成REPORT是本次实际运行记录，可继续补充实验分析和答辩说明。

## 物理与网络约定

- 正式输入直接使用JSON中的左右眼c2w，单位米，不从文件名解析位姿，不重建look-at矩阵。
- 精确有效屏宽0.596736m、高0.335664m，三层x=+0.06/0/−0.06m，水平FOV40°。
- EyeReal预训练模型按“右眼在前、屏幕远→近”的顺序训练（`lego_bulldozer`数据中`_0`为右眼）。适配层把网络输入重排为[右,左]和远→近，每张图始终与自己的pose配对；输出再转换成助教L1/L2/L3的近→远顺序。计分、渲染与导出仍按[左,右]。2026-11-06修正前误用[左,右]输入，30步结果为42.18dB，修正后为46.27dB。
- 网络初值裁剪至[0,1]，Projected Adam只优化三层像素，量化后选检查点；最终使用真实RGB8图样。
- 新GPU renderer缓存双精度逐射线几何，硬mask遵循助教规则。默认用它对导出的RGB8图样自测（与助教NumPy前向最大误差<1e-7、mask一致，公开样例分数逐位相同），避免每例约5–12秒的NumPy前向；加`--reference-score`改用未修改的助教NumPy函数计分。PNG眼图只用于预览。
- 时间预算按实测的每例导出开销和网络/几何开销为剩余样例预留，并为最后打包预留15秒，剩余时间平均分给各例精修。

## 清单数据用于后续训练

`ExamDataset`读取助教示范或自行采集的相同JSON/JSONL格式，按真实索引读取每个样例，验证RGB8图、c2w和0.06m眼距。返回`images, views_m, screens_m`，眼序为网络使用的[右,左]（图与pose配对），屏幕顺序为远→近；输出交给TA renderer前须翻转为近→远。采集时的场景中心、朝向、尺度仍要由采集配置明确记录。

### 训练接入时的顺序约定

这是后续接入训练时需要遵守的接口约定，当前推理调用链没有重复重排的逻辑错误。`ExamPredictor`从TA输入构造[左,右]图像与位姿，再交给`network_initial`转换；当前没有把`ExamDataset`的返回值直接传给`network_initial`的调用。

| 接口 | 图像与对应位姿的顺序 | 屏幕顺序 | 调用方式 |
|---|---|---|---|
| 助教清单 | left/pose_l、right/pose_r | TA约定近→远 | 作为外部输入保存，不改清单语义 |
| `network_initial`的输入 | 左→右 | 近→远 | 由函数内部转换成网络顺序，返回近→远图样 |
| `ExamDataset`的输出 | 右→左 | 远→近 | 已完成转换，直接送入`EyeRealNet` |
| 直接调用`EyeRealNet`的输出 | 图样没有左右眼轴 | 远→近 | 进入TA前向或导出前，转换为近→远 |

`ExamDataset`与`network_initial`是两种接入路径，不能将数据集的返回值直接再套一遍`network_initial`；那样才会产生眼序和层序的二次翻转。数据集类注释中的“Same input order as network_initial”应理解为最终送入网络时的约定一致，不能理解为两种接口接收相同顺序。

以下仅示意数据集路径的模型前向。`model`须已按训练需要配置，三个输入须已放到相同设备；完整训练入口还需设置梯度、损失、优化器和验证流程。

```python
images, views_m, screens_m = dataset[index]  # 右→左，屏幕远→近
patterns_back_to_front = model(
    images[None], views_m[None], screens_m[None]
)[0]
patterns_front_to_back = patterns_back_to_front[[2, 1, 0]]
```

后续物理前向的眼图顺序由传入的位姿顺序决定：使用上述`views_m`时，预测眼图也为[右,左]，应与同顺序的`images`计算损失。需要按TA接口计分或导出眼图时，再将眼图与对应位姿一起转换回[左,右]。不要只交换图像或只交换位姿。

## 环境与检查

本机实测为Windows、Python3.12、PyTorch2.11.0+cu128、torchvision0.26.0+cu128；无需更改已有环境。随附requirements保留助教Python3.10兼容的NumPy/Pillow版本，并加入所用Torch版本。Linux x86_64可沿用`bash create_env.sh`和`bash inference.sh ...`，其新建环境流程未在当前Windows主机上运行。

```powershell
& 'D:\anaconda\envs\test\python.exe' .\testing\check_submission.py --index .\testing\demo_input\test.json --submission ..\group_new\patterns.json
& 'D:\anaconda\envs\test\python.exe' .\tools\verify_ta_alignment.py
& 'D:\anaconda\envs\test\python.exe' -m unittest lightfield_lab.test_exam -v
```

计时记录包括main入口后的模型初始化、输入输出、求解、NumPy参考计分、格式检查与打包，排除进程启动及main之前的导入。600秒是含输出预留的软分配，不是操作系统强制截止；以本次运行`within_time_limit`与最终考试通知为准。

隐藏深度指标未提供；`demo/index.html`只展示当前图样与重建，不代表已获得专项加分。历史实验以legacy几何回放，新的解析实验默认使用ta几何，见主目录对齐报告。


默认模式（2026-11-07 经 H100 复核）：`--iteration-dtype float32 --geometry-backend torch`。float32 用于迭代，Torch 在当前设备生成 float64 几何缓存；量化后检查点仍按 float64 评分。回放原基线需显式使用 `--iteration-dtype legacy --geometry-backend numpy`。`float64` 模式则把参数、采样与迭代损失全部设为 float64。步数和学习率仍为 30、0.02，尚未进入 T1-3 选参。验证见 `docs/T1复核与默认切换_20261107.md`。


## 快速输出与离轴研究入口

`fast_inference.py` 支持同样的求解参数，使用有界I/O线程池、不生成非必需预览，仍等待图样写盘、格式检查、代码和权重打包后才完成计时；输出包内的启动脚本指向快速入口。N=100/300的实测见 ../docs/速度精度与量化实测_20261107.md。

`offaxis_inference.py --intrinsics <K清单>` 为独立研究入口，固定输入图片和外参，输出完整K与图样元数据；官方离轴格式未知，不能把研究清单当作已对齐的正式提交格式。见 ../docs/离轴建模与实验_20261107.md。
