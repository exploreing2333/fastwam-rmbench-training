# FastWAM RMBench 独立训练项目

本项目保留 `rmbench_fastwam_gt_cover_current_gbs64_200k` 当前代码、实际环境和保存的训练配置，为 RMBench 的 12 个任务分别训练模型。GitHub 保存代码、配置、数据处理程序和任务元数据；权重、完整环境、原始数据、GT 标注和训练数据从 [魔搭 ModelScope](https://modelscope.cn/datasets/ccenzl/fastwam-rmbench-training-assets) 下载。默认从公共 Wan + ActionDiT 预训练初始化开始，`resume=null`。

## 一行下载并恢复环境

在克隆后的项目目录执行：

```bash
bash setup.sh
```

命令将完整资源下载到配置要求的相对目录，并恢复 Python 环境与 CUDA 编译工具链。支持断点续传；新下载的文件检查 SHA256。资源约 257 GB，环境解包额外占用约 15 GB，训练输出另计。魔搭资源仓库公开，下载无需 Token，也无需安装 ModelScope SDK。远端存在 `ASSETS_READY.json` 才表示整套资源已上传并验证完成。

目标服务器需要 Linux x86_64、Python 3.8 或以上、bash、tar、gzip、flock，以及可运行 CUDA/BF16 的 NVIDIA 驱动。无需预先安装 conda 或训练依赖。

恢复后完整环境使用 Python 3.10.20、PyTorch 2.7.1+cu128、DeepSpeed 0.18.5、Accelerate 1.12.0、Transformers 4.57.1。CUDA 编译工具链为 12.1，沿用本次参考训练的实际环境。完整版本见 `environment/installed_packages.json` 和 `environment/requirements.lock.txt`。

环境与运行缓存默认位于 `.runtime/`。项目放在对象存储或网络挂载目录时，可以指定本机高速磁盘：

```bash
FASTWAM_RUNTIME_ROOT=/scratch/fastwam_rmbench_runtime bash setup.sh
```

后续训练需使用相同的 `FASTWAM_RUNTIME_ROOT`，也可以提前 `export FASTWAM_RUNTIME_ROOT=/scratch/fastwam_rmbench_runtime`。

## 每个任务一行启动

每条命令默认使用 8 张 GPU，训练 200000 optimizer steps，global batch 为 64。首次训练命令也会自动完成资源下载与环境恢复。

参考任务 `cover_blocks` 以外的 11 个任务：

```bash
bash train_task.sh press_button
bash train_task.sh battery_try
bash train_task.sh place_block_mat
bash train_task.sh classify_blocks
bash train_task.sh rearrange_blocks
bash train_task.sh swap_blocks
bash train_task.sh storage_blocks
bash train_task.sh blocks_ranking_try
bash train_task.sh put_back_block
bash train_task.sh observe_and_pickup
bash train_task.sh swap_T
```

重新训练参考任务：

```bash
bash train_task.sh cover_blocks
```

逐条选择需要的任务启动即可。按顺序运行全部 11 个其他任务、同一时刻只训练一个任务：

```bash
bash train_tasks_sequential.sh
```

默认单任务后台运行，关闭终端后继续。日志和权重写到 `runs/TASK/RUN_ID/`，主日志为 `stdout.log`，PID/进程组分别记录在 `training.pid` 和 `training.pgid`。使用 `--foreground` 可在当前终端运行。

4 卡仍保持 global batch=64，启动器自动改成 microbatch 8、accumulation 2：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_task.sh rearrange_blocks --gpus 4
```

仅检查配置和启动命令，不开始 GPU 训练：

```bash
bash train_task.sh rearrange_blocks --dry-run
```

`--num-workers 2` 可减少每个 GPU 进程的数据加载 worker；默认 8，沿用参考配置。`--gradient-checkpointing` 可显式启用显存优化；默认关闭。`--cached-stats` 可使用按相同原代码计算的任务归一化统计；默认仍重新计算。

## 资源目录

| 目录 | 内容 |
| --- | --- |
| `data/RMBench/` | 完整原始数据、GT 标注、pi05 处理数据、转换与采集代码、模拟资源 |
| `data/lerobot/rmbench_all_12tasks_demo_clean_50/` | 原训练使用的完整 LeRobot 缓存，600 episodes、416711 frames |
| `data/gt_subtask_text_embeds/` | 46 个 GT subtask 对应的 Wan T5 文本缓存 |
| `models/` | 公共 Wan 2.2、VAE、T5、tokenizer 与 ActionDiT 初始化 |
| `environment/` | 完整 conda-pack 环境、CUDA toolkit，以及实际依赖版本 |
| `vendor/` | 原环境依赖的 curobo、LIBERO 源码与资源 |
| `assets/manifest.json` | 外部资源路径、大小、SHA256；下载后保持同一目录布局 |
| `data/task_manifest.json` | 每任务 50 条 episode 的索引、原始编号及帧数 |

自训练权重与训练状态不在迁移资源中。训练依赖 GT subtask，初始化时无需加载训练过的 V72/ProgressModel。

## 保留的训练与采样逻辑

- Seed 42；8 GPU × microbatch 8 × accumulation 1 = global batch 64；BF16；AdamW fused；LR 1e-5；weight decay 0.01；max grad norm 1。
- Warmup 10000 steps，之后 cosine 至 200000 steps。视频与动作 scheduler 的 train/infer shift 均为 5。
- 每 2000 steps 保存 checkpoint，保留最近 2 个；`eval_every=0`，`wandb=false`。
- 每任务使用参考缓存中对应的 50 条 episode，保留原始顺序。绝对 qpos 动作维度 14、proprio 维度 14，使用原 z-score 归一化方法。
- 输入仅包含当前 RGB 三视角和当前 proprio，没有历史帧、在线进度模型或数值进度。相机拼接为 384×320 RobotWin mosaic。
- 从当前帧预测 32 个动作，视频采样为 t,t+4,...,t+32，共 9 帧。
- GT 来自原始 `language_annotation.json`，按原始逐段时长展开，再截到 LeRobot 长度。`state[t]` 与原始 qpos[t]、`action[t]` 与原始 qpos[t+1] 精确对齐。
- 当前 GT subtask 构造固定 Wan prompt，使用 128×4096 BF16 T5 context；padding context 置零，context mask 保持全部可见。
- 每个 GT segment 独立处理。跨 subtask 的动作后缀重复本段最后动作；视频后缀重复本段最后已采样帧，继续参与 loss。
- 所有合法真实帧均可作为 anchor，包括短 segment 和尾部。每 epoch 按 seed+epoch 无放回打乱；DDP 对齐使用 -1 零损失样本，恢复时精确跳过已消费的 global microbatch。

迁移只改变任务选择、资源路径、输出路径和恢复状态；模型架构、优化器、loss、图像处理与采样算法沿用原代码。

## 代码与验证记录

训练入口为 `scripts/train.py`；模型与训练循环位于 `src/fastwam/runtime.py` 和 `src/fastwam/trainer.py`。GT subtask 数据集为 `src/fastwam/datasets/lerobot/gt_subtask_video_dataset.py`，分段与 terminal hold 在 `progress_subtask_video_dataset.py`，采样顺序在 `src/fastwam/utils/samplers.py`。

`data/RMBench/` 保留数据转换代码。`scripts/build_rmbench_raw_source_map.py` 构造原始数据对应关系，`scripts/compute_task_stats.py` 使用原算法计算归一化，`scripts/build_gt_text_cache.py` 可从公共 T5 离线重建 GT 文本缓存。启动时根据当前项目位置重建资源路径，因此无需原服务器目录。

`provenance/` 保留来源版本、训练代码差异和文件 SHA256；`reports/` 保留验证结果：12 任务、600 episodes 的标量/GT/sampler 检查，24 episodes 的 623 次实际 RGB/action/context 取样，以及完整环境迁移与启动配置校验。打包和验证均未启动新的 GPU 训练。

完整资源额外校验会读取约 257 GB：`bash download_assets.sh --check --verify`。共享本地存储可用 `bash setup.sh --source /path/to/complete/snapshot --link-local`，或用 `--source /path/to/complete/snapshot` 复制资源。

原 curobo 源码中的 224 个网格文件是尚未展开的 Git LFS 引用，GT 训练不读取它们。`assets/manifest.json` 保留完整原引用，下载时原样恢复，避免将占位引用误当成模型或训练数据上传。
