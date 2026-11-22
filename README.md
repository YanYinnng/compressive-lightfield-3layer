# 三层压缩光场显示：逐例物理求解、离轴相机建模与深度分层

**Compressive light field display with three stacked LCD layers: per-sample physical solving, off-axis camera modelling, and depth layering.**

这是上海创智学院金秋营实训课题"AI 驱动的压缩光场物理建模和成像分析"的个人解决方案（单人组，2026-11-05 至 11-08）。

最终正式测试的四项排名分：

| 画质（50） | 速度（10） | 分层（20） | 离轴（20） |
|---|---|---|---|
| **50** | **10** | 10 | 16 |

- 正式测试共 1400 例，全流程 214 s；
- 自测双眼 PSNR：第一问 55.7 dB，离轴 55.0 dB。

## 问题

- **显示器**：三层 LCD，屏宽 0.596736 m，1920×1080。三层分别位于 L2 前后 6 cm（x = +0.06 / 0 / −0.06 m）。
- **成像模型**：一条视线穿过三层，三层相位相加后得到亮度，I = sin²(π/2·Σgₖ)。三层是相位叠加，没有遮挡。
- **输入与输出**：输入一对左右眼图像和两只眼的位姿，输出两套三层 RGB8 图样：
  - 第一问：在标准相机下，两只眼看到的画面要接近输入图；
  - 离轴：自己声明每只眼的离轴相机 K 和 pose，评测时复用原图。

一条视线只给出一个方程：g₁ + g₂ + g₃。两只眼的方程比三层共 3N 个未知数少得多，因此存在大量**等价解**：许多组三层图样在两只眼里看起来完全一样。四项评分于是对应四个不同的问题：

| 评分项 | 要回答的问题 | 评测怎么做 |
|---|---|---|
| 画质 | 两只眼看得对不对 | 直接比目标图 |
| 速度 | 时间花在哪 | 分段计时 |
| 离轴 | 相机怎么定义 | 独立的评分相机 |
| 分层 | 在等价解里挑哪一个 | 双眼 PSNR 不敏感，需自建分层指标 + 目视 |

## 方法

### 画质：在精确前向上逐例求解
- 网络（EyeReal 公开权重）一次前向只有约 33 dB。我们把它的输出当初值，再在精确前向模型上逐例做投影 Adam（100 步，lr 0.08），按实际提交的 RGB8 量化结果选检查点。最终做到 55–57 dB。
- 渲染器在 GPU 上重写、可求导，与官方前向函数逐像素核对：最大误差 ≤ 6×10⁻⁸，有效掩码零差异。所以自测分数就是官方分数。

### 速度：拆解耗时构成，一项一项抠

test3 规模（1100 例）的全流程耗时：

| 阶段 | 耗时 | 发现 |
|---|---|---|
| 原始 PyTorch 实现 | 1168 s | Adam 约占单例耗时的 60%：每步要搬运约 9 GB 数据，实际只需约 0.7 GB |
| + 融合 CUDA 算子 | 420 s | GPU 求解快了约 7 倍，端到端却只快 2.8 倍 |
| + 自写 PNG 编码 | 约 200 s | 容器只有 20 核 CPU 配额，写 PNG 抢占了驱动 GPU 的线程 |
| + 两例在两条 CUDA 流上重叠 | 约 186 s | GPU 剩下的约 3 成空闲被填满 |

- **融合算子**（`lightfield_lab/fused.py`）：
  - 针孔眼看平面层，眼像素到层像素的映射是一个单应，两眼三层共 54 个数，坐标现算，不缓存采样网格。
  - 每步只用两个内核。K1 负责前向、残差和梯度。K2 用逆单应 gather 出双线性采样的严格转置，不用原子操作、结果确定，并在同一内核里完成 Adam 和截断。
  - 每步 3.4 → 0.43 ms。
- **PNG 编码**（`lightfield_lab/png_fast.py`）：Up 滤波 + ISA-L deflate，每张 136 → 15 ms，无损往返已验证。
- 各阶段的输出都与上一版逐位一致。

### 离轴：难点在建模，也就是怎样定义离轴观看的相机，而不在求解

![离轴相机对比](docs/figures/offaxis_camera_comparison.gif)

- **为什么要建离轴相机**：标准相机的画框和屏幕没有关系，屏幕只占画面的一部分，斜看时还是个梯形。离轴相机要让"相机要看的画面"和"屏幕这扇窗"对齐。
- **模型**：
  - 相机中心在眼睛；
  - 成像面就是 L2 屏幕，图像像素与屏幕像素一一对应；
  - 主点是眼睛在 L2 上的垂足；
  - 焦距 = 眼到 L2 的垂距 × 每米像素数。

  写成矩阵：K = [[s·pₓ, 0, W/2 + s·p_y], [0, s·pₓ, H/2 − s·p_z], [0, 0, 1]]，s = 1920 / 0.596736 像素/米。这就是 Kooima 广义离轴投影。
- **为什么只改 K 不够**：把 L2 四角和光心相连得到视锥，用垂直于原光轴的平面去截，得到的是梯形。把梯形变成矩形需要透视变换，而 K 的最后一行固定为 [0, 0, 1]，放不下透视项，只能放进旋转 R；眼睛位置不变。
- **交叉评测**（lego 100 对，dB）：

  | 求解 \ 评分 | 主点平移 | L2 屏幕相机 |
  |---|---|---|
  | 主点平移 | 56.24 | 13.48 |
  | L2 屏幕相机 | 14.07 | 51.62 |

  同一个求解器，相机定义对了 51.6 dB，错了只有 13.5 dB。用自己的相机求解、再用同一个相机自评，是循环论证。
- **复用原图仍然自洽**：两只眼共用同一块成像面，复用原图 51.4 dB，按该相机重新渲染 51.8 dB。
- **求解**：离轴分支用无训练的偏振场层析 SART 作初值（`lightfield_lab/tomographic.py`），再做 Adam 精修。

[交互式图解](docs/offaxis_camera_tutorial.html)可以拖动眼睛位置，对比四种相机建法。需要下载到本地后用浏览器打开。

### 分层：在等价解中挑选

![分层对比](docs/figures/layering_lego.jpg)

*lego 三层图样，从左到右依次为 L1、L2、L3。上排是原方案：三层都是整车拷贝。下排是分层版：L3 是亮背景配暗剪影，L1、L2 以物体的轮廓和细节为主。*

- **根因**：
  - 只按画质精修，一条视线上三层采样相位的梯度相同，修正被平分到三层，于是长出三份拷贝；
  - 想给某层加权补救，Adam 按像素归一化，梯度乘固定权重后更新几乎不变，加权失效。
- **方案**（`lightfield_lab/mono_depth.py`、`fused.py` 的加权模式）：
  1. 用 Depth Anything V2 Small 估相对深度；
  2. 按前、中、后景计算每个层像素对 L1/L2/L3 的归属；
  3. 加权 SART 按归属分配残差，不再平分；
  4. 再用 Adam 补回画质。
- **效果**（lego，层内亮度与归属权重的相关，0 为随机）：

  | 方案 | 层内相关 |
  |---|---|
  | 原方案 | −0.04 |
  | 提交方案（单目估深） | 约 0.14 |
  | 真值深度（上限参照） | 0.63 |

- **局限**：两只眼各自做单目估计再取平均，得到的相对深度很不可靠，也没用上双眼视差。下一步应该用双眼视差标定深度，并让分层约束贯穿整个精修。

### 评测复盘
几乎每次进步都来自评测体系的改变：
- 离轴换了评分相机，才发现自评高分靠不住；
- 速度改成端到端计时，才找到写图的瓶颈；
- 分层把三张图并排看，才发现指标涨了，三层也可能还是拷贝。

做法总结为四条：
1. 先与官方口径对齐；
2. 用好、坏两种参照校准指标；
3. 指标加目视；
4. 评测与求解解耦。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `lightfield_lab/` | 核心库 |
| `lightfield_lab/exam_geometry.py` | 与官方对齐的米制渲染器 |
| `lightfield_lab/fused.py` | 融合 CUDA 求解器（CuPy/NVRTC，或 torch 扩展） |
| `lightfield_lab/tomographic.py` | SART 层析 |
| `lightfield_lab/offaxis_models.py`、`intrinsics.py` | 离轴相机模型 |
| `lightfield_lab/mono_depth.py` | 单目深度与层归属 |
| `lightfield_lab/png_fast.py` | 快速 PNG 编码 |
| `lightfield_lab/exam_adapter.py`、`v3_submission.py` | 逐例求解流程 |
| `lightfield_lab/test_*.py` | 单元测试 |
| `solution/submit_v3.py` | 一条命令生成完整提交：两套图样、报告、代码、格式检查 |
| `solution/run_formal.sh` | 解包测试集、求解、检查 |
| `tools/` | 实验与基准脚本：离轴交叉评测、速度剖析、PNG 编码对比、焦栈与深度评测等 |
| `testing_v3/` | 课题方提供的 v3 前向参考与格式检查器（AGPL-3.0，未修改） |
| `third_party/depth_anything_v2_src/` | Depth Anything V2 源码（Apache-2.0，未修改） |
| `EyeReal/` | 上游 EyeReal（git 子模块，AGPL-3.0） |
| `docs/` | 离轴交互图解与插图 |

## 运行

```bash
git clone --recursive https://github.com/YanYinnng/compressive-lightfield-3layer.git
cd compressive-lightfield-3layer
pip install -r solution/requirements_v3.txt   # 另需 CUDA 版 PyTorch、CuPy；python-isal 可选
python tools/fetch_weights.py                 # EyeReal 预训练权重 -> weights/pretrained_model.pth
# Depth Anything V2 Small：从 Hugging Face depth-anything/Depth-Anything-V2-Small
# 下载 depth_anything_v2_vits.pth 放到 weights/

python solution/submit_v3.py --index <test_input 目录或 test.json> --out group_example
# 默认：--solver fused --layering mono --layering-eps 0.3 --lanes 2 --offaxis-model l2_camera
python testing_v3/check_submission.py --index <test.json> --submission group_example/index.json
```

- **依赖**：Python 3.10+、PyTorch（CUDA）、torchvision、NumPy、Pillow、SciPy、CuPy。没有 CuPy 时，可设置 `LF_FUSED_BACKEND=torch`，用 nvcc 编译 torch 扩展。
- **测试**：`python -m unittest discover -s lightfield_lab -p "test_*.py" -t .`。
  - 本地 52 项中 44 项通过；
  - 另有 7 项需要课题方的 v1.4 资料包（`testing/`、演示输入、lego 3DGS 资产），本仓库不包含这些，所以会报找不到文件。

**说明**：本仓库不包含课题方的数据集、练习集和 3DGS 资产，也不包含模型权重和实验原始结果。评测平台专用的提交与调度脚本也已移除。

## 许可

- 本仓库代码以 **AGPL-3.0** 发布（见 `LICENSE`），因为它派生自 EyeReal（AGPL-3.0）。
- 第三方组件的许可见 `solution/THIRD_PARTY_NOTICES.md`。
