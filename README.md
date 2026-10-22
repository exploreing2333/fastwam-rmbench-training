# FastWAM RMBench 独立训练项目

本项目保留 `rmbench_fastwam_gt_cover_current_gbs64_200k` 当前代码、实际环境和保存的训练配置，为 RMBench 的 12 个任务分别训练模型。GitHub 保存代码、配置、数据处理程序和任务元数据；权重、完整环境、原始数据、GT 标注和训练数据从 [魔搭 ModelScope](https://modelscope.cn/datasets/ccenzl/fastwam-rmbench-training-assets) 下载。每次启动默认创建新 run，从 **Wan 2.2 预训练权重初始化**开始训练。只有显式加 `--resume` 才查找并恢复本任务最近保存完成的完整训练状态。

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

### 完整 instruction 训练入口（12 个任务）

每个任务均可加 `--instruction-mode full`，使用 LeRobot 中该任务原始的完整 instruction。
训练样本仍由当前帧及未来 32 步 action、9 帧视频组成；窗口在同一个 episode 内连续取值，
可以跨过原 subtask 边界，不再读取 subtask 标签或做 subtask 末尾保持，仅使用原版 episode 末尾填充及 mask。
每个 epoch 对全部真实帧做 `torch.randperm(seed + epoch)` 随机打乱，每帧出现一次；
DDP 对齐使用零损失样本，resume 精确跳过已消费的 global microbatch。

```bash
bash train_task.sh press_button --instruction-mode full
bash train_task.sh battery_try --instruction-mode full
bash train_task.sh cover_blocks --instruction-mode full
bash train_task.sh place_block_mat --instruction-mode full
bash train_task.sh classify_blocks --instruction-mode full
bash train_task.sh rearrange_blocks --instruction-mode full
bash train_task.sh swap_blocks --instruction-mode full
bash train_task.sh storage_blocks --instruction-mode full
bash train_task.sh blocks_ranking_try --instruction-mode full
bash train_task.sh put_back_block --instruction-mode full
bash train_task.sh observe_and_pickup --instruction-mode full
bash train_task.sh swap_T --instruction-mode full
```

该模式的 run、checkpoint 和日志位于 `runs/<task>/full_instruction/<run_id>/`。
默认新建 run 从头训练；显式 `--resume` 只选择该任务完整 instruction 模式最近保存完成的状态，
并拒绝混用 subtask 模式 checkpoint。示例：

```bash
bash train_task.sh press_button --instruction-mode full --resume
bash train_task.sh press_button --instruction-mode full --dry-run
bash train_tasks_sequential.sh --instruction-mode full
```

完整 instruction 模式的顺序启动命令默认包含全部 12 个任务，也可用 `--tasks` 指定子集。
启动时自动校验 `${FASTWAM_RUNTIME_ROOT}/full_instruction_text_embeds/`，缺失的文本缓存使用
随项目提供的 Wan T5 权重在 CPU 编码一次；沿用 128×4096 BF16 context 和原版 padding mask 行为。
instruction 和数据窗口之外的任务设置沿用对应 `portable_<task>.yaml`，包括
`observe_and_pickup` 的首帧 init-KV、global batch=64、200000 steps、每 2000 steps 保存和保留最近两个 checkpoint。
不加 `--instruction-mode full` 时继续使用下述 subtask 入口。

真实样本及随机采样审查程序：`scripts/audit_full_instruction_training.py`。

### observe_and_pickup：轨迹初始帧 + 原始 subtask 标签

`configs/portable_observe_and_pickup.yaml` 现使用与当前 Swap T 相同的 init-KV 方案。
每条轨迹始终额外输入该轨迹第 0 帧的完整三相机 VAE 信息（48×24×20，patch 后 120 个 token），
经可训练 Video DiT 在全部 30 层提供 K/V，供当前视觉和 action 分支注意力读取。
使用共享视频 3D RoPE：init 时间为 0，当前帧与 action 的时间为轨迹绝对控制帧索引 / 16。
训练时保留梯度，只在同一 batch 内复用相同轨迹及 subtask 的 init 分支；不跨 optimizer step 缓存 K/V。

标签明确读取 `data/RMBench/data/data/observe_and_pickup/demo_clean/language_annotation.before_target_labels.json`，
指令恢复为 `Wait for wall appear.` 和 `Pick up the target object.`。现有目标描述版 JSON 保留归档。
原有 subtask 分段、逐帧采样、32 步 action、9 帧视频目标、归一化、损失、LR、每卡 BS=8、
8 卡 global BS=64、200000 步均保持原设置；没有加入其他历史帧或冻结额外参数。

```bash
git pull --ff-only
bash train_task.sh observe_and_pickup
```

这条训练命令默认从头开始，并自动检查当前资源版本、补齐魔搭数据和原始标签；无需另跑 `setup.sh`。
魔搭的 `RESOURCE_MANIFEST.json` 中 `task_training_inputs.observe_and_pickup` 登记了当前使用的原始标签路径和校验值。
50 条轨迹的原始 RGB/action 与训练缓存沿用已校验的原始数据，历史目标描述标签不进入当前训练。

启动器在空闲 GPU 检查通过后，自动生成/校验 50 条轨迹的首帧 VAE 缓存，位于
`${FASTWAM_RUNTIME_ROOT}/init_frame_latents/observe_and_pickup/`。VAE 只需一次离线编码，
训练时直接读取 latent；`--dry-run` 仅生成配置预览。旧的 current-only / 目标描述标签 checkpoint
不能作为此配置的完整状态 resume；后续此方案自身的 checkpoint 可以正常 `--resume`。

审查脚本为 `scripts/audit_observe_init_frame.py`；旧的 `audit_observe_target_labels.py` 专门检查历史目标描述版资源。

每条命令默认从头训练，使用 8 张 GPU，总目标为 200000 optimizer steps，global batch 为 64。加 `--resume` 续训时自动沿用 checkpoint 的 GPU 数、microbatch 和梯度累积。首次训练命令也会自动完成资源下载与环境恢复。

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

从头训练参考任务：

```bash
bash train_task.sh cover_blocks
```

逐条选择需要的任务启动即可。按顺序运行全部 11 个其他任务、同一时刻只训练一个任务：

```bash
bash train_tasks_sequential.sh
```

默认单任务后台运行，关闭终端后继续。日志和权重写到 `runs/TASK/RUN_ID/`，主日志为 `stdout.log`，PID/进程组分别记录在 `training.pid` 和 `training.pgid`。使用 `--foreground` 可在当前终端运行。

### 默认从头训练，显式加 --resume 续训

上述所有单任务命令及 `bash train_tasks_sequential.sh` 均默认创建新 run，从 Wan 初始化开始训练；已有 checkpoint 会保留。加 `--resume` 可自动选择最新完整 checkpoint，无需手填路径：

```bash
# 默认从头训练，即使本任务已有 checkpoint
bash train_task.sh put_back_block
# 从本任务最新完整 checkpoint 续训
bash train_task.sh put_back_block --resume
```

使用 `--resume` 时，启动器仅检查 `runs/put_back_block/*/checkpoints/state/step_*`，验证 `trainer_state.json`、模型、optimizer 分片、scheduler 和各 rank RNG 文件。多个 run 按完成标记的保存时间选择最新状态，同时间再按 step 排序；未完成或缺少文件的状态会跳过并打印原因。续训恢复模型、optimizer、scheduler、global step 和数据采样位置，默认复用原 run 目录，继续每 2000 步保存、最多保留最近两个 checkpoint。旧启动配置归档到 `launch_history/`，日志追加写入。`--dry-run` 将预览写入 `dry_runs/`，不会覆盖原启动配置。

使用 `--resume` 且所选完整状态已经达到 `--max-steps`（默认 200000）时，该任务直接完成，不启动 GPU 进程；顺序训练会继续下一个任务。不加 `--resume` 时，即使旧 run 已完成，也从头训练。`--max-steps` 始终表示总步数。加 `--resume` 但没有任何 checkpoint 时从头训练；仅有独立 `.pt` 权重或全部状态不完整时会明确报错。独立 `.pt` 没有 optimizer、scheduler 或采样状态。

```bash
# --fresh 保留为默认从头训练行为的显式写法
bash train_task.sh put_back_block --fresh
# 指定完整状态目录
bash train_task.sh put_back_block --resume runs/put_back_block/RUN_ID/checkpoints/state/step_118000
# 全部队列从各自最新状态恢复；完成的任务自动跳过
bash train_tasks_sequential.sh --resume
```

使用 `--resume` 时，若显式指定 GPU 数、microbatch 或梯度累积，必须与所选完整状态一致。`--fresh` 也可用于顺序训练命令，与 `--resume` 互斥。

首次使用 4 卡仍保持 global batch=64，启动器自动改成 microbatch 8、accumulation 2：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_task.sh rearrange_blocks --gpus 4
```

仅检查配置和启动命令，不开始 GPU 训练：

```bash
bash train_task.sh rearrange_blocks --dry-run
```

`--num-workers 2` 可减少每个 GPU 进程的数据加载 worker；首次训练默认 8，续训未指定时沿用原 run 的数量。`--gradient-checkpointing` 可显式启用显存优化；首次训练默认关闭，续训保留原 run 已启用的选项。`--cached-stats` 可使用按相同原代码计算的任务归一化统计；默认仍重新计算。

## 迁移服务器检查与 CPFS 报错修复

`multiprocessing.resource_sharer → SocketListener.bind → OSError: [Errno 95] Operation not supported` 表示临时目录所在文件系统不能建立 Unix socket。旧版运行时会把终端设置的 `TMPDIR` 覆盖为 `.runtime/cache/tmp`，因此仅在终端执行 `export TMPDIR=/tmp` 不足以解决 CPFS 上的报错。

当前启动器在导入 PyTorch/Accelerate 前选择并测试 IPC 临时目录，默认使用 `/tmp/fastwam-ipc-UID`，同时处理 `tempfile` 与 multiprocessing 各自的目录缓存。需要自定义时可设置 `FASTWAM_IPC_TMPDIR=/tmp/fastwam-ipc`；该目录必须可写、路径较短且支持 Unix socket。Hugging Face、Arrow、Torch、Triton 等文件缓存继续保存在 `${FASTWAM_RUNTIME_ROOT}/cache/`，与 IPC 临时目录分别配置。环境路径在进入项目工作目录前转换为绝对路径。

已有服务器更新代码即可复用现有魔搭资源：

```bash
git pull --ff-only
```

如果之前已手改 `src/fastwam/runtime.py`，导致上述更新被本地修改阻止，先将该文件保存到 Git stash，再更新；保留 stash 供后续审查，避免直接恢复旧的 TMPDIR 覆盖逻辑：

```bash
git stash push -m 'runtime-before-portability-fix' -- src/fastwam/runtime.py
git pull --ff-only
```

重复启动前，先列出当前项目的旧训练作业：

```bash
bash stop_training.sh --list
bash stop_training.sh --run-dir runs/observe_and_pickup/RUN_ID --stop --force
```

将 `RUN_ID` 替换为列表中需要停止的失败作业。若明确要停止当前项目列出的全部训练，可使用 `bash stop_training.sh --all --stop --force`。脚本按当前项目的训练入口、run 目录及进程启动时间识别进程；进程仍受 `strace` 等调试器跟踪时会报告 `TracerPid`，需先检查并关闭对应的旧跟踪器，再重新停止作业。

在目标服务器验证 8 卡、两 worker 与真实任务数据：

```bash
bash check_runtime.sh --gpus 8 --num-workers 2 --real-data --task observe_and_pickup --output .runtime/runtime_check.json
```

检查内容包括实际导入的代码位置、Unix socket、两 worker 的张量传输、文件缓存写入/重命名/锁、实际 Arrow 缓存位置、每张 GPU 的 BF16 前向/反向与 fused AdamW、小规模真实数据读取，以及现有作业和共享内存容量。输出 `RUNTIME_CHECK_PASSED` 后再开始训练；这验证目标服务器当时的环境，完整模型能否运行仍以实际 optimizer step 为准。

若 `/dev/shm` 容量不足，可扩大容器共享内存，或将检查和训练命令中的 `--num-workers 2` 改为 `--num-workers 0`。零 worker 保留 global batch、阶段构造及训练配置，数据读取在训练主进程中进行。首次恢复后可先前台检查两步训练，再启动正式 200000 步任务：

```bash
bash train_task.sh observe_and_pickup --gpus 8 --num-workers 2 --foreground --max-steps 2
```

正常完成两步后，去掉 `--foreground --max-steps 2` 启动正式训练。启动器自动执行上述环境预检，保存 `runtime_preflight.json`，并对当前项目占用重叠 GPU 的旧作业进行检查；启动锁防止两个命令同时重复启动。复查结果见 `reports/runtime_all_tasks_data_audit.json`、`reports/runtime_activation_audit.json` 与 `tests/test_runtime_portability.py`。数据、预训练权重、模型、loss、归一化及采样规则沿用当前配置。

## 资源目录

| 目录 | 内容 |
| --- | --- |
| `data/RMBench/` | 完整原始数据、GT 标注、pi05 处理数据、转换与采集代码、模拟资源 |
| `data/lerobot/rmbench_all_12tasks_demo_clean_50/` | 原训练使用的完整 LeRobot 缓存，600 episodes、416711 frames |
| `data/gt_subtask_text_embeds/` | 102 个已登记 GT subtask 的 Wan T5 文本缓存，包括原始 observe_and_pickup、归档的目标描述和按键逐次编号标签 |
| `models/` | 公共 Wan 2.2、VAE、T5、tokenizer，以及由 Wan 权重插值生成的动作骨干初始化 |
| `environment/` | 完整 conda-pack 环境、CUDA toolkit，以及实际依赖版本 |
| `vendor/` | 原环境依赖的 curobo、LIBERO 源码与资源 |
| `assets/manifest.json` | 外部资源路径、大小、SHA256；下载后保持同一目录布局 |
| `data/task_manifest.json` | 每任务 50 条 episode 的索引、原始编号及帧数 |

自训练权重与训练状态不在迁移资源中。训练依赖 GT subtask，初始化时无需加载训练过的 V72/ProgressModel。

## 从 Wan 初始化

默认启动所有任务时，视频 DiT 直接加载 `Wan-AI/Wan2.2-TI2V-5B` 的预训练权重。动作 DiT 的骨干通过 `scripts/preprocess_action_dit_backbone.py` 从同一 Wan 权重生成：将对应张量线性插值到动作骨干的 1024 hidden dim，并按原算法执行 alpha scaling。

`models/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` 保存的是上述 Wan 权重转换结果，仅包含动作骨干；动作输入层 `action_encoder`、动作输出层 `head` 和 `proprio_encoder` 在训练启动时按 seed 42 新建。该文件不包含 RMBench 训练后的动作模型、动作头或 optimizer 状态。

配置模板保留 `skip_dit_load_from_pretrain=false` 和上述 Wan 动作骨干转换路径。启动器默认将 `resume` 设为 `null`；只有显式加 `--resume` 才选择本任务最新完整训练状态或指定的状态目录。初始化来源核对结果见 `reports/wan_initialization_audit.json`。

## V72 subtask 分段

所有任务默认使用 `gt_subtask_segment_mode: v72_semantic`。训练读取 GT 标注时，仅合并相邻且文本在统一大小写、空白后相同的段，累计它们的时长；被其他指令隔开的同名段分别保留。阶段编号、GT 文本列表及动作/视频的末尾保持边界一起更新。`gt_subtask_event_instance_tasks: [press_button]` 保留每一次按键事件。

`press_button` 的 50 条演示共 448 次按压。每次按压都有独立标签，例如 `Press the middle button for the third time.`；左键和中键各有第 1–9 次标签，确认键沿用原文。完整的目标次数、当前序号和持续帧数保存在 `data/RMBench/data/data/press_button/demo_clean/press_count_labels.json`，原标注保存在同目录的 `language_annotation.before_count_labels.json`。启动训练时，`press_button_label_mode: ordinal_per_press_v1` 会校验标签顺序和序号。

目标总次数只作为数据元信息保存，不进入当前模型的文本条件。环境每次按压后复位按钮，已按次数只存在环境内部；当前模型仅看一帧三视角 RGB 和关节状态。若将同一按钮的一组按压合并为固定的“按 N 次”subtask，组末与下一次按压前可能出现近乎相同的输入，却要求不同动作。保留逐次边界也保证 32 步动作和视频目标在边界处保持于本次按压。若今后让目标总次数进入模型，在线验证必须从可观测输入提供同样的数值，并增加可用的已完成次数或历史条件。

| 任务 | 每条轨迹原始段数 → 默认 V72 段数 |
| --- | --- |
| `classify_blocks` | 24 → 2 |
| `rearrange_blocks` | 11 → 3 |
| `swap_blocks` | 14 → 5 |
| `storage_blocks` | 16–24 → 6 或 8 |
| `put_back_block` | 10 → 3 |
| `observe_and_pickup` | 4 → 2 |
| `swap_T` | 7 → 4 |

`cover_blocks`、`battery_try`、`place_block_mat`、`blocks_ranking_try` 的阶段边界和 `press_button` 的独立事件保持原有行为。V72 合并操作保留各帧原有图像、动作和归一化统计；`press_button` 与下面的 `observe_and_pickup` 更新了 GT 文本及对应 T5 缓存。

需要复现原始 GT 分段时，可显式选择：

```bash
bash train_task.sh classify_blocks --subtask-segments raw_annotation
```

默认从 Wan 初始化开始训练。分段改变会更新受影响任务的 sampling contract；显式加 `--resume` 续训旧状态时沿用原训练器的合同变更处理，重新开始完整的数据 epoch。完整核对见 `reports/v72_semantic_subtask_audit.json`。

### 按键段的填充 action loss 消融

以下四个任务通过 `exclude_terminal_hold_loss_subtasks` 启用填充区 action loss 屏蔽：

| 任务 | 精确匹配的子任务标签 |
| --- | --- |
| `put_back_block` | `Press the button.` |
| `rearrange_blocks` | `Press button.` |
| `blocks_ranking_try` | `Press the button.` |
| `press_button` | 左键和中键各第 1～9 次的 18 种完整编号标签，例如 `Press the left button for the first time.` |

这些指定子任务使用 `action_loss_mask=action_real_mask`；同一任务的其他阶段为全真 mask。32 步 action 的数据填充、GT 合并边界、视频目标、真实停顿和采样顺序保持原有逻辑。人工补出的 action 后缀不参与 action loss；分母仍为 32，不按剩余真实长度重新放大尾部样本。设置空列表可恢复原监督。`press_button` 的 `Press the confirm button.` 和 `swap_blocks` 保留原监督。

更新代码后运行原启动命令将从头训练；需要恢复完整训练状态时加 `--resume`。两种方式均使用当前任务配置中的 loss 屏蔽列表。此次 loss 更新不改变数据和文本标签，不需要重新下载资源；续训时保留原采样位置。

四个启用任务及 `swap_blocks` 对照任务的全部 250 条标注已核对资源清单 SHA256，覆盖 696 个被屏蔽填充监督的按键段；标签和边界 mask 检查见 `reports/rmbench_selected_press_hold_loss_validation_20261015.json`。

2026-10-15 对全部 600 条轨迹、416711 帧的核对中，资源清单 SHA256、原始关节目标与 LeRobot action 的下一帧对应关系全部通过。按键段的实际终点如下，比例为人工填充占该类按键段全部 action 目标的比例：

| 任务/按键类型 | 段数 | 段长度 | 人工填充占比 | 填充重复的实际姿态 |
| --- | ---: | ---: | ---: | --- |
| put_back_block | 50 | 109–111 | 14.13% | 按后抬起并撤回原位 |
| rearrange_blocks | 50 | 106–116 | 13.92% | 按后抬起并撤回原位 |
| blocks_ranking_try | 198 | 48–77 | 27.60% | 按后抬起约 3.82 cm |
| press_button 左键/中键 | 398 | 47–76 | 28.92% | 按后抬起约 3.82 cm |
| press_button 确认键 | 50 | 99–100 | 15.59% | 压住按钮 |
| swap_blocks | 50 | 65–76 | 23.62% | 压住按钮 |

前三类及左键/中键具有类似的阶段歧义，现已全部启用取消人工填充监督的消融；填充比例本身不能证明失败原因，收益仍需训练和验证确认。`swap_blocks` 和确认键的填充支持保持按压，保留原监督。逐帧结果见 `reports/rmbench_all_tasks_padding_audit_20261015.json`，按键终点结果见 `reports/rmbench_press_endpoints_audit_20261015.json`。

## observe_and_pickup 历史目标描述标注（归档）

历史版本曾为全部 50 条轨迹新增目标物体标注，覆盖 19 种外观描述。当前训练使用前述 init-KV + 原始标签方案；下列资源保留用于复查历史实验。例如灰色鼠标轨迹的历史两个阶段为：

```text
Observe the medium gray computer mouse on the shelf and wait for the wall to appear.
Pick up the matching medium gray computer mouse from the table.
```

该历史版本按 V72 规则将原始 4 段合并成观察、拾取两个阶段，各段时长与动作/视频末尾保持边界沿用原标注，新增 38 个文本缓存。当前默认仍按 V72 分段，`raw_annotation` 可显式改为原始 4 段，但两者均读取当前配置指定的原始通用标签。

新标注位于 `data/RMBench/data/data/observe_and_pickup/demo_clean/language_annotation.json`，目标元数据及旧标注备份一并保存在魔搭资源中。GitHub 中的 `provenance/observe_and_pickup_target_labels.json` 保留每条轨迹的目标描述、模型编号和标注来源。图像、动作数据、归一化统计和其他 11 个任务的标注可复用。

历史验证结果见 `reports/observe_target_labels_audit.json`；可用 `python scripts/audit_observe_target_labels.py --rgb-samples` 复查目标描述版资源。当前 init-KV 配置使用 `scripts/audit_observe_init_frame.py --rgb-samples` 审查。

## 保留的训练与采样逻辑

- Seed 42；8 GPU × microbatch 8 × accumulation 1 = global batch 64；BF16；AdamW fused；LR 1e-5；weight decay 0.01；max grad norm 1。
- Warmup 10000 steps，之后 cosine 至 200000 steps。视频与动作 scheduler 的 train/infer shift 均为 5。
- 每 2000 steps 保存 checkpoint，保留最近 2 个；`eval_every=0`，`wandb=false`。
- 每任务使用参考缓存中对应的 50 条 episode，保留原始顺序。绝对 qpos 动作维度 14、proprio 维度 14，使用原 z-score 归一化方法。
- 基础输入包含当前 RGB 三视角和当前 proprio；`observe_and_pickup` 额外读取轨迹第 0 帧的完整 VAE latent 并注入 K/V。相机拼接为 384×320 RobotWin mosaic。
- 从当前帧预测 32 个动作，视频采样为 t,t+4,...,t+32，共 9 帧。
- GT 默认来自 `language_annotation.json`；`observe_and_pickup` 显式读取 `language_annotation.before_target_labels.json`。按上述 V72 规则构造阶段，再截到 LeRobot 长度。`state[t]` 与原始 qpos[t]、`action[t]` 与原始 qpos[t+1] 精确对齐。
- 当前 GT subtask 构造固定 Wan prompt，使用 128×4096 BF16 T5 context；padding context 置零，context mask 保持全部可见。
- 每个 V72 GT 阶段独立处理。跨真正 subtask 边界的动作后缀重复本阶段最后动作；视频后缀重复本阶段最后已采样帧。除上述显式配置的 action loss 消融外，填充继续参与 loss。合并段内部的原始边界继续使用真实未来动作和视频。
- 所有合法真实帧均可作为 anchor，包括短 segment 和尾部。每 epoch 按 seed+epoch 无放回打乱；DDP 对齐使用 -1 零损失样本，恢复时精确跳过已消费的 global microbatch。

优化器、图像处理和每个 epoch 的真实帧采样顺序沿用原代码；`observe_and_pickup` 使用前述 init-KV 注入。GT 阶段按上述 V72 规则合并；action loss 保留显式配置的按键段填充监督消融。

## 代码与验证记录

训练入口为 `scripts/train.py`；模型与训练循环位于 `src/fastwam/runtime.py` 和 `src/fastwam/trainer.py`。GT subtask 数据集为 `src/fastwam/datasets/lerobot/gt_subtask_video_dataset.py`，分段与 terminal hold 在 `progress_subtask_video_dataset.py`，采样顺序在 `src/fastwam/utils/samplers.py`。

`data/RMBench/` 保留数据转换代码。`scripts/build_rmbench_raw_source_map.py` 构造原始数据对应关系，`scripts/compute_task_stats.py` 使用原算法计算归一化，`scripts/build_gt_text_cache.py` 可从公共 T5 离线重建 GT 文本缓存。启动时根据当前项目位置重建资源路径，因此无需原服务器目录。

`provenance/` 保留原始来源版本、训练代码差异和文件 SHA256；`reports/gt_metadata_audit.json` 记录初始原始分段的标量/GT/sampler 验证。更新后的阶段、文本、动作/视频保持及采样顺序验证见 `reports/v72_semantic_subtask_audit.json`，真实 RGB/action/context 取样见 `reports/gt_rgb_sample_audit.json`，启动配置见 `reports/github_launch_audit.json`。打包和验证均未启动新的 GPU 训练。

完整资源额外校验会读取约 257 GB：`bash download_assets.sh --check --verify`。共享本地存储可用 `bash setup.sh --source /path/to/complete/snapshot --link-local`，或用 `--source /path/to/complete/snapshot` 复制资源。

原 curobo 源码中的 224 个网格文件是尚未展开的 Git LFS 引用，GT 训练不读取它们。`assets/manifest.json` 保留完整原引用，下载时原样恢复，避免将占位引用误当成模型或训练数据上传。
