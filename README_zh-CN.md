<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_logo.png" width="500"/>
</div>

## 简介

[English](/README.md) | 简体中文

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/zh_CN/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

MMAction2 是一款基于 PyTorch 的视频理解开源工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一

主分支代码目前支持 **PyTorch 1.3 以上**的版本

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_overview.gif" width="380px"><br>
    <p style="font-size:1.5vw;">Kinetics-400 上的动作识别</p>
  </div>
  <div style="float:right;margin-right:0px;">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px"><br>
    <p style="font-size:1.5vw;">NTURGB+D-120 上的基于人体姿态的动作识别</p>
  </div>
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/spatio-temporal-det.gif" width="800px"/><br>
    <p style="font-size:1.5vw;">AVA-2.1 上的时空动作检测</p>
</div>

## 主要特性

- **模块化设计**：MMAction2 将统一的视频理解框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的视频理解模型

- **支持多种任务和数据集**：MMAction2 支持多种视频理解任务，包括动作识别，时序动作检测，时空动作检测以及基于人体姿态的动作识别，总共支持 **27** 种算法和 **20** 种数据集

- **详尽的单元测试和文档**：MMAction2 提供了详尽的说明文档，API 接口说明，全面的单元测试，以供社区参考

## 更新记录

- (2021-11-24) 在 NTU60 XSub 上支持 **2s-AGCN**， 在 joint stream 和 bone stream 上分别达到 86.06% 和 86.89% 的识别准确率。
- (2021-10-29) 支持基于 skeleton 模态和 rgb 模态的时空动作检测和行为识别 demo (demo/demo_video_structuralize.py)。
- (2021-10-26) 在 NTU60 3d 关键点标注数据集上训练测试 **STGCN**, 可达到 84.61% (高于 [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17135) 中的 81.5%) 的识别准确率。
- (2021-10-25) 提供将 NTU60 和 NTU120 的 3d 骨骼点数据转换成我们项目的格式的脚本(tools/data/skeleton/gen_ntu_rgbd_raw.py)。
- (2021-10-25) 提供使用自定义数据集训练 PoseC3D 的 [教程](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/custom_dataset_training.md)，此 PR 由用户 [bit-scientist](https://github.com/bit-scientist) 完成！
- (2021-10-16) 在 UCF101, HMDB51 上支持 **PoseC3D**，仅用 2D 关键点就可分别达到 87.0% 和 69.3% 的识别准确率。两数据集的预提取骨架特征可以公开下载。

v0.21.0 版本已于 2021 年 12 月 31 日发布，可通过查阅 [更新日志](/docs/changelog.md) 了解更多细节以及发布历史

## 安装

请参考 [安装指南](/docs_zh_CN/install.md) 进行安装

## 教程

请参考 [基础教程](/docs_zh_CN/getting_started.md) 了解 MMAction2 的基本使用。MMAction2也提供了其他更详细的教程:

- [如何编写配置文件](/docs_zh_CN/tutorials/1_config.md)
- [如何微调模型](/docs_zh_CN/tutorials/2_finetune.md)
- [如何增加新数据集](/docs_zh_CN/tutorials/3_new_dataset.md)
- [如何设计数据处理流程](/docs_zh_CN/tutorials/4_data_pipeline.md)
- [如何增加新模块](/docs_zh_CN/tutorials/5_new_modules.md)
- [如何导出模型为 onnx 格式](/docs_zh_CN/tutorials/6_export_model.md)
- [如何自定义模型运行参数](/docs_zh_CN/tutorials/7_customize_runtime.md)

MMAction2 也提供了相应的中文 Colab 教程，可以点击 [这里](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial_zh-CN.ipynb) 进行体验！

## 模型库

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">行为识别方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/c3d/README_zh-CN.md">C3D</a> (CVPR'2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README_zh-CN.md">TSN</a> (ECCV'2016)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README_zh-CN.md">I3D</a> (CVPR'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README_zh-CN.md">I3D Non-Local</a> (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/r2plus1d/README_zh-CN.md">R(2+1)D</a> (CVPR'2018)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/trn/README_zh-CN.md">TRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README_zh-CN.md">TSM</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README_zh-CN.md">TSM Non-Local</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowonly/README_zh-CN.md">SlowOnly</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowfast/README_zh-CN.md">SlowFast</a> (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/csn/README_zh-CN.md">CSN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tin/README_zh-CN.md">TIN</a> (AAAI'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tpn/README_zh-CN.md">TPN</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/x3d/README_zh-CN.md">X3D</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/omnisource/README_zh-CN.md">OmniSource</a> (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition_audio/resnet/README_zh-CN.md">MultiModality: Audio</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tanet/README_zh-CN.md">TANet</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/timesformer/README_zh-CN.md">TimeSformer</a> (ICML'2021)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时序动作检测方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/ssn/README_zh-CN.md">SSN</a> (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/bsn/README_zh-CN.md">BSN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/bmn/README_zh-CN.md">BMN</a> (ICCV'2019)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时空动作检测方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/acrn/README_zh-CN.md">ACRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/ava/README_zh-CN.md">SlowOnly+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/ava/README_zh-CN.md">SlowFast+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/lfb/README_zh-CN.md">LFB</a> (CVPR'2019)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">基于骨骼点的动作识别方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/stgcn/README.md">ST-GCN</a> (AAAI'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/README.md">PoseC3D</a> (ArXiv'2021)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

各个模型的结果和设置都可以在对应的 config 目录下的 *README_zh-CN.md* 中查看。整体的概况也可也在 [**模型库**](https://mmaction2.readthedocs.io/zh_CN/latest/recognition_models.html) 页面中查看

MMAction2 将跟进学界的最新进展，并支持更多算法和框架。如果您对 MMAction2 有任何功能需求，请随时在 [问题](https://github.com/open-mmlab/mmaction2/issues/19) 中留言。

## 数据集

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="4" style="font-weight:bold;">动作识别数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/hmdb51/README_zh-CN.md">HMDB51</a> (<a href="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/">主页</a>) (ICCV'2011)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101/README_zh-CN.md">UCF101</a> (<a href="https://www.crcv.ucf.edu/research/data-sets/ucf101/">主页</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/README_zh-CN.md">ActivityNet</a> (<a href="http://activity-net.org/">主页</a>) (CVPR'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README_zh-CN.md">Kinetics-[400/600/700]</a> (<a href="https://deepmind.com/research/open-source/kinetics/">主页</a>) (CVPR'2017)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/sthv1/README_zh-CN.md">SthV1</a> (<a href="https://20bn.com/datasets/something-something/v1/">主页</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/sthv2/README_zh-CN.md">SthV2</a> (<a href="https://20bn.com/datasets/something-something/">主页</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/diving48/README_zh-CN.md">Diving48</a> (<a href="http://www.svcl.ucsd.edu/projects/resound/dataset.html">主页</a>) (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/jester/README_zh-CN.md">Jester</a> (<a href="https://20bn.com/datasets/jester/v1">主页</a>) (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/mit/README_zh-CN.md">Moments in Time</a> (<a href="http://moments.csail.mit.edu/">主页</a>) (TPAMI'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/mmit/README_zh-CN.md">Multi-Moments in Time</a> (<a href="http://moments.csail.mit.edu/challenge_iccv_2019.html">主页</a>) (ArXiv'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/hvu/README_zh-CN.md">HVU</a> (<a href="https://github.com/holistic-video-understanding/HVU-Dataset">主页</a>) (ECCV'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/omnisource/README_zh-CN.md">OmniSource</a> (<a href="https://kennymckormick.github.io/omnisource/">主页</a>) (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README_zh-CN.md">FineGYM</a> (<a href="https://sdolivia.github.io/FineGym/">主页</a>) (CVPR'2020)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">时序动作检测数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/thumos14/README_zh-CN.md">THUMOS14</a> (<a href="https://www.crcv.ucf.edu/THUMOS14/download.html">主页</a>) (THUMOS Challenge 2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/README_zh-CN.md">ActivityNet</a> (<a href="http://activity-net.org/">主页</a>) (CVPR'2015)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">时空动作检测数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101_24/README_zh-CN.md">UCF101-24*</a> (<a href="http://www.thumos.info/download.html">主页</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/jhmdb/README_zh-CN.md">JHMDB*</a> (<a href="http://jhmdb.is.tue.mpg.de/">主页</a>) (ICCV'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ava/README_zh-CN.md">AVA</a> (<a href="https://research.google.com/ava/index.html">主页</a>) (CVPR'2018)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">基于骨骼点的动作识别数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-FineGYM</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-NTURGB+D</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-UCF101</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-HMDB51</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
  </tr>
</table>

标记 * 代表对应数据集并未被完全支持，但提供相应的数据准备步骤。整体的概况也可也在 [**数据集**](https://mmaction2.readthedocs.io/en/latest/supported_datasets.html) 页面中查看

## 基准测试

为了验证 MMAction2 框架的高精度和高效率，开发成员将其与当前其他主流框架进行速度对比。更多详情可见 [基准测试](/docs_zh_CN/benchmark.md)

## 数据集准备

请参考 [数据准备](/docs_zh_CN/data_preparation.md) 了解数据集准备概况。所有支持的数据集都列于 [数据集清单](/docs_zh_CN/supported_datasets.md) 中

## 常见问题

请参考 [FAQ](/docs_zh_CN/faq.md) 了解其他用户的常见问题

## 相关工作

目前有许多研究工作或工程项目基于 MMAction2 搭建，例如：

- Evidential Deep Learning for Open Set Action Recognition, ICCV 2021 **Oral**. [[论文]](https://arxiv.org/abs/2107.10161)[[代码]](https://github.com/Cogito2012/DEAR)
- Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective, ICCV 2021 **Oral**. [[论文]](https://arxiv.org/abs/2103.17263)[[代码]](https://github.com/xvjiarui/VFS)
- Video Swin Transformer. [[论文]](https://arxiv.org/abs/2106.13230)[[代码]](https://github.com/SwinTransformer/Video-Swin-Transformer)

更多详情可见 [相关工作](docs/projects.md)

## 许可

该项目开源自 [Apache 2.0 license](/LICENSE)

## 引用

如果你觉得 MMAction2 对你的研究有所帮助，可以考虑引用它：

```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```
