# Smart Photo Narrative v2

这是一个面向个人照片的本地多模态检索、事件组织与可信叙事系统。当前版本复用 `torchtest` Conda 环境中已有的 CLIP 和 BLIP，不需要重新引入一套视觉模型；Qwen Scene Graph 因本机显存不足，设计为在 AIRE 上离线生成后再安全导回。

系统主链路如下：

```text
个人照片
  -> CLIP 视觉语义 + BLIP Caption + Qwen Scene Graph + EXIF 时间/地点
  -> CLIP / BM25 / Scene Graph 独立召回 + Metadata 预过滤 + RRF 融合
  -> 文本检索 / 以图搜图 / 日期与关系查询 / before-after 查询
  -> 确定性事件组织
  -> 带照片证据引用的回忆、日记或故事
```

## 当前真实数据状态

下表为合并第二批照片 (pic2) 后 `scripts/report_index_state.py` 的实测状态（最近核对 2026-08-02），不是示例数字：

| 项目 | 数量 | 说明 |
|---|---:|---|
| SQLite 照片记录 | 1599 | 376 主相册 + 1223 pic2 第二批 |
| CLIP 图像向量 | 1599 | Chroma `photo_image_v2` |
| BLIP Caption / FTS5 文档 | 1599 | 同时写入 SQLite FTS5 |
| CLIP 索引标记照片 | 1599 | 非语义哨兵 `__clip_indexed__`，承载 CLIP 完成/版本身份；不进 FTS、不进 Story 证据 |
| Scene Graph（成功） | 1599 | 376（主相册）+ 1223（pic2）；均为身份/哈希绑定的 legacy 恢复 |
| Scene Graph 三元组 / 向量 | 22230 | SQLite 与 Chroma `photo_scene_graph_v2` 计数一致 |
| 自动事件 | 226 | 1579 张照片已分配，20 张低时间置信度照片暂不强制归入事件 |
| 含 GPS 照片 | 317 | — |
| 索引失败 | 0 | — |

演进：2026-07-16 完成 376 张主相册 v2 全量基线（约 9 分 02 秒）；2026-07-17 恢复入库 374 个个人 Scene Graph；2026-07-25 合并 pic2 第二批 1223 张照片 + Scene Graph（增量 CLIP+BLIP 约 28 分 46 秒、0 失败；pic2 与主相册有 1 张字节完全相同的重复图，已按内容哈希去重）。增量快路对无变化照片跳过 SHA-256、EXIF 和模型推理。以上时间仅代表本机这一批数据，不应当作其他设备或数据集的性能结论。

2026-07-27 移除了 72 个 CLIP 零样本预设标签（§35）：语义标签不再生成，`tags` 表只保留每张照片一行非语义哨兵以承载 CLIP 完成/版本身份标记（增量快路与 v1→v2→v1 保护不变）；标签同时退出 Story 证据与 Creative 第一人称资格判定。检索排序不受影响（标签从不参与打分，冻结 qrels 无标签过滤）。

**重要边界：已冻结的 A0–A4 v1 检索消融与 N0–N3 / Qwen / Mood 等 Story 正式实验，均在 2026-07-16 的 376 张相册上完成，未在扩充后的 1599 张相册上重跑。** 扩充后的相册用于功能演示与后续实验；下文各实验节的正式结论仍以 376 张 case study 为准。pic2 批次同样是 legacy 恢复，标记 `generation_sha256_verified=false`，不声称生成时 Manifest 哈希验证。

## 运行环境

本项目已经在 `torchtest` 中使用以下核心组件：

- Python 3.10
- PyTorch 2.10.0 + CUDA 12.6
- Transformers 5.2.0
- CLIP：`openai/clip-vit-base-patch32`
- BLIP：`Salesforce/blip-image-captioning-base`
- ChromaDB 1.5.1
- Streamlit 1.54.0
- SQLite FTS5（由 Python 内置 SQLite 提供）

本机为 RTX 3050 Ti Laptop GPU 4 GB。默认 CLIP 批次为 8、BLIP 批次为 4；发生 CUDA OOM 时会清理缓存并将当前批次减半，最低降到 1。CLIP 阶段完成并卸载后才进入 BLIP 阶段，避免两个模型同时长期占用显存。普通批次失败还会降为逐张处理，并记录失败项和检查点。

已有 `torchtest` 环境时，直接使用它即可，不要为了启动项目重复安装 CLIP、BLIP 或覆盖现有 CUDA 版 PyTorch。只有重建新环境时才参考 [requirements.txt](requirements.txt)，并先按机器 CUDA 版本安装匹配的 PyTorch。

## 快速启动

从项目根目录运行：

```powershell
cd "E:\leeds\smart-photo-narrative"
conda run -n torchtest python -m streamlit run app.py
```

浏览器通常会打开 `http://localhost:8501`。页面提供相册与索引、文本/图片检索、事件、故事、Scene Graph 工作流和诊断信息。

Ollama 是可选能力，不是启动条件，也不要求为了使用本项目先测试 Ollama：

- 默认规则解析器可以处理普通、日期、关系及 before-after 查询；
- 生产环境默认使用本地 `qwen3:4b`，所有聊天请求显式设置 `think=False`；
- 勾选后才会尝试使用同一 Qwen 模型辅助查询解析；
- 故事生成可调用 Qwen，但不可用或输出未通过证据校验时，会退回带照片引用的确定性文本；
- CLIP、BLIP、FTS5、Chroma、事件组织和 Scene Graph 导入均不依赖 Ollama。

模型可通过环境变量 `SMART_PHOTO_OLLAMA_MODEL` 覆盖；默认值和上下文设置见
[config.py](config.py)。已经冻结的 N0--N3 正式实验仍在
[story_evaluation.py](story_evaluation.py) 中固定使用原 `llama3:latest` 及其 digest，生产模型切换不会改写既有实验结果。

## 索引：增量、强制重算与状态报告

照片放在 `photos/`。索引以文件内容 SHA-256 和稳定 `photo_id` 为身份依据，改名不依赖 basename 匹配。SQLite 是规范数据源，Chroma 是可重建的向量派生索引。

日常增量索引只处理新增、内容变化或缺少阶段结果的照片，并在完成后重新组织事件：

```powershell
conda run -n torchtest python scripts/rebuild_multimodal_index.py --organize-events
```

无变化快路先比较精确相对路径、文件大小、mtime 和 ctime，再复用已保存的 SHA-256 与 EXIF；向量、当前 CLIP 身份标签和指定版本 Caption 仍会分别检查并按缺失阶段回补。CLIP 重算前先撤销旧的 `clip:*` 完成标记，向量写成功后再事务性写入当前身份和 FTS，避免版本切换或中途失败时旧标签与新向量混用。元数据临时失败不会写入“完整”指纹，强制重算失败也会保留为下一轮待重试状态。文件系统时间戳不是密码学证明：极端情况下，同路径、同大小且人为保留所有时间戳的内容替换可能逃过日常快路，因此正式实验冻结数据前应运行一次 `--force` 强校验并保存最终状态报告。

需要对所有已发现照片重新计算 CLIP 和 BLIP 时使用 `--force`：

```powershell
conda run -n torchtest python scripts/rebuild_multimodal_index.py --force --organize-events
```

`--force` 表示重算所有照片的模型结果，不等于删除原照片，也不会删除旧版 Chroma 备份。脚本结束时会输出 `FINAL_REBUILD_REPORT=...`，其中含索引计数、失败数、耗时、向量计数和事件统计，可直接保存到实验日志。

仅查看当前 SQLite/Chroma 一致性，不运行 CLIP 或 BLIP：

```powershell
conda run -n torchtest python scripts/report_index_state.py
```

报告包括照片、Caption、FTS5、CLIP 索引标记、Scene Graph、事件、故事、失败项、GPS、时间来源、最新索引运行和 Chroma collection 数量。

## 数据库与版本化路径

| 用途 | 当前路径 |
|---|---|
| v2 SQLite 规范库 | `data/smart_photo.db` |
| v2 Chroma 向量库 | `indexes/chroma_v2/` |
| 原 v1 Chroma 路径 | `chroma_db/` |
| 重建前冻结备份 | `chroma_db_legacy_20260716_pre_multimodal/` |

SQLite 使用迁移、外键、WAL 和 FTS5，保存照片身份与哈希、EXIF/文件时间置信度、Caption、标签、Scene Graph 与三元组、事件、故事、索引运行和失败记录。Chroma 1.5.1 中分别维护版本化的图像向量 collection 和 Scene Graph 三元组 collection。

不要把 `chroma_db/` 或冻结备份复制覆盖到 `indexes/chroma_v2/`。旧库用于回滚与对照，v2 的 SQLite 和 Chroma 应作为一组检查。

## 检索如何工作

一次标准文本检索按以下顺序进行：

1. 先在 SQLite 根据日期、标签、位置、时间置信度或事件得到合格 `photo_id` 集合；
2. CLIP 对查询与图像向量做视觉语义召回；
3. SQLite FTS5 对 BLIP Caption 等文本做 BM25 召回；
4. Chroma 对每条 Qwen Scene Graph 三元组独立召回；
5. 三路结果用加权 Reciprocal Rank Fusion 融合，而不是把所有内容提前拼成一段文本；
6. 返回每张照片的命中通道、通道排名、Caption 词和 Scene Graph 三元组解释。

当前权重为 CLIP `1.0`、Caption/BM25 `0.8`、Scene Graph `0.9`，RRF 常数为 `60`。Metadata 是召回前的约束条件，不是另一份向量，因此能明确区分“语义相关”和“时间/地点条件满足”。

支持的任务包括：

- 自然语言文本检索；
- 上传查询图片进行以图搜图；
- 日期范围、标签、地点、最低时间置信度和事件过滤；
- 主体—关系—客体式关系查询；
- “A 之前/之后发生了什么”的时间邻居查询；
- “A 之后出现 B”一类有顺序、同一天且受时间窗口约束的 before-after 成对查询；
- 根据日期、时间间隔、CLIP 相似度和 GPS 距离组织事件。

before-after 的顺序由可靠时间戳验证，不会仅因两张图语义相似就判为时间关系。低置信度文件时间不会被当作与可靠 EXIF 等价的证据。

## 可信叙事

故事可以基于单张照片、选中的照片篮子、日期或已保存 Event 生成。Story v3 先把照片转换为带稳定编号的模型观察，再按 Event、时间和相似度组织成最多 5 个证据组；CLIP 近重复照片压缩观察但保留全部证据编号，BLIP/Qwen 明显冲突移到不确定观察区。

多图故事按证据组而非按照片生成段落，每个段落必须引用对应组内证据。Faithful 默认模式还校验中英文、引用、组覆盖、重复、第一人称及目的/情绪/因果推测；失败最多修复一次，之后使用按组聚合的确定性回退。只有用户填写的 `verified_context` 可支持个人身份或背景事实。引用意味着“这段文字由哪些照片/模型观察支持”，并不把 BLIP、CLIP 或 Qwen 的预测提升为人工确认的事实。

Ollama 仅用于可选的语言生成或查询解析。没有 Ollama 时，用户仍可完成所有检索、事件组织和证据回退故事流程。

## Qwen Scene Graph：本地导出、AIRE 三 GPU、严格导入

个人照片不能在本机 4 GB GPU 上运行 Qwen2.5-VL-7B，因此采用离线批次。旧 `qwen_scene_graph_outputs_full.jsonl` 属于 Geograph 数据；即使文件名相同，只要完整 SHA-256 不同就不能导入个人相册。

### 1. 本地生成身份与哈希绑定的批次

Scene Graph 导出只接受实际内容为 JPEG 的文件。HEIC、PNG 等应先用 `scripts/convert_photos_to_jpg.py` 进行有备份和校验的规范化。

```powershell
conda run -n torchtest python scripts/export_scene_graph_batch.py `
  --album-root photos `
  --export-dir outputs/scene_graph_batch `
  --dataset-id personal-main-v1
```

输出包括 `manifest.jsonl` 与 `images/<photo_id>.jpg`。Manifest 固定包含 schema、`dataset_id`、`photo_id`、相对路径、SHA-256 和文件大小；导出图片保持 JPEG 字节不变。

### 2. 上传 AIRE 并提交三 GPU 数组任务

按照仓库上一级的 `aire_hpc_upload_guide.md` 上传代码与 `outputs/scene_graph_batch/`。在 AIRE 项目目录运行：

```bash
mkdir -p /mnt/scratch/$USER/smart-photo-narrative/logs
mkdir -p /mnt/scratch/$USER/smart-photo-narrative/outputs
sbatch scripts/cloud/run_scene_graph_array_3gpu.sbatch
```

Slurm 配置为数组 `0-2%3`，每个任务只申请一张 GPU，分别输出：

```text
qwen_scene_graph_shard_0.jsonl
qwen_scene_graph_shard_1.jsonl
qwen_scene_graph_shard_2.jsonl
```

默认采用已验证的 Qwen2.5-VL-7B 参数：vLLM `max-model-len=1536`、生成 `max_tokens=640`、每图最多 15 条三元组。Checkpoint 只跳过身份、哈希、模型和 prompt 版本都一致的成功非空行；失败、空结果和坏行会在重跑时继续处理。脚本只连接计算节点上的本地 vLLM，默认 `OPENAI_API_KEY=EMPTY`，不要将真实密钥写入代码、Manifest、JSONL 或 Slurm 脚本。

### 3. 本地先审计，再显式生成可入库 JSONL

下载三个 shard 后先 dry-run；引号中的通配符由脚本展开：

```powershell
conda run -n torchtest python scripts/import_scene_graph_results.py `
  --manifest outputs/scene_graph_batch/manifest.jsonl `
  --album-root photos `
  --results "outputs/qwen_scene_graph_shard_*.jsonl"
```

只有 `dataset_id + photo_id + manifest SHA-256 + 当前本地文件 SHA-256` 全部一致、来源为 `remote` 或 `remote_repaired`、状态成功且三元组非空的行才可进入下一步。系统从不按 basename 或 stem 接受结果。

确认审计报告后显式写出干净结果：

```powershell
conda run -n torchtest python scripts/import_scene_graph_results.py `
  --manifest outputs/scene_graph_batch/manifest.jsonl `
  --album-root photos `
  --results "outputs/qwen_scene_graph_shard_*.jsonl" `
  --apply `
  --output-jsonl outputs/scene_graph_indexable.jsonl `
  --report-json outputs/scene_graph_import_report.json
```

### 4. 预演并写入 SQLite/Chroma

审计脚本不会自动改变主索引。先预演映射与冲突：

```powershell
conda run -n torchtest python scripts/index_scene_graph_results.py `
  outputs/scene_graph_indexable.jsonl --dry-run
```

确认 `checksum_conflicts` 和 `malformed` 均为 0 后再正式入库：

```powershell
conda run -n torchtest python scripts/index_scene_graph_results.py `
  outputs/scene_graph_indexable.jsonl
```

这里会再次用完整内容哈希把云端身份映射到 SQLite 的稳定 UUID，保存规范化三元组，并用现有 CLIP 文本编码器将每条三元组写入 Scene Graph collection。完成后运行 `scripts/report_index_state.py` 核对数据库和向量数。

更详细的云端步骤见 [scripts/cloud/README_scene_graph_aire.md](scripts/cloud/README_scene_graph_aire.md)。

### 5. 已确认但缺少旧 Manifest 的个人批次恢复

正常流程仍必须优先使用以上严格 Manifest。只有用户能够确认旧结果确实来自当前、未经修改的个人相册时，才允许使用独立恢复入口；普通 legacy importer 的个人相册禁令不会因此放宽。

先只读 dry-run：

```powershell
conda run -n torchtest python scripts/recover_legacy_personal_scene_graphs.py `
  outputs/scene_graph/raw/qwen_scene_graph_photos_full.jsonl `
  --album-root photos `
  --database data/smart_photo.db `
  --dataset-id personal-main-v1-legacy-recovered-20260717 `
  --output-jsonl outputs/scene_graph/recovered/scene_graph_indexable_recovered_20260717.jsonl `
  --report-json outputs/scene_graph/recovered/scene_graph_recovery_report_20260717.json `
  --confirm-generated-from-current-album
```

确认 `blocking_issues=0` 后追加 `--apply` 生成恢复 JSONL，再按第 4 节进行索引 dry-run 和正式入库。恢复记录固定写入 `provenance=legacy_path_time_recovered`、原始结果文件哈希、当前照片哈希、SQLite UUID、行哈希和验证条件，同时明确 `generation_sha256_verified=false`。这是一条诚实记录证据缺口的兼容路线，不能在论文中描述为生成时 Manifest 哈希验证。

检索器中的 `caption` 通道只查询 FTS5 的 BLIP Caption 列，绝不读取同一 FTS 文档中的 Scene Graph、Tags 或 Location；否则 A1 会泄漏 A2 的信息。通用 `PhotoStorage.search_bm25(..., fields=...)` 仍可在非消融场景显式选择这些字段。

## A0–A4 检索消融评测

评测定义固定为：

| 变体 | 启用内容 |
|---|---|
| A0 | CLIP |
| A1 | CLIP + BLIP Caption/BM25 |
| A2 | CLIP + Qwen Scene Graph |
| A3 | CLIP + Caption/BM25 + Scene Graph |
| A4 | A3 + qrels 中明确给出的 Metadata 条件 |

冻结的 qrels v1 含 48 条查询：六类各 8 条、36 英文/12 中文、12 dev/36 test。私有查询、UUID 和地点保存在 Git 忽略目录，公开仓库只保存 [标注协议](evaluation/QRELS_FORMAT.md)、类别统计和 SHA-256。两轮打乱顺序的 agent 标注后，用户于 2026-07-20 终审正例与疑难项，纠正 1 个错误关系正例；当前 48 条均为 `human_validated`。系统没有把自身排名直接当作真值，正式 Recall、MRR 或 nDCG 只能由后续 A0–A4 运行产生。

运行完整 A0–A4：

```powershell
conda run -n torchtest python scripts/evaluate_retrieval.py `
  evaluation/my_qrels.jsonl `
  --output outputs/retrieval_ablation_report.json `
  --ks 1 5 10 `
  --repeats 3 `
  --warmup 1
```

只运行指定变体：

```powershell
conda run -n torchtest python scripts/evaluate_retrieval.py `
  evaluation/my_qrels.jsonl --variants A0 A1 A3 A4
```

before-after 是独立时间任务，不混入 A0–A4。只有 qrels 含 `task_type: temporal_pair` 且相关项使用 `before_photo_id->after_photo_id` 时，才运行：

```powershell
conda run -n torchtest python scripts/evaluate_retrieval.py `
  evaluation/my_qrels.jsonl --include-temporal-mode
```

Ollama 解析器默认关闭以保证可复现性；只有确实要评测它时才额外传 `--ollama-parser`。报告记录 qrels SHA-256、运行环境、每个查询的排名、Recall@K、MRR、nDCG@K、P50/P95 延迟与重复运行一致性。

完整标注格式见 [evaluation/QRELS_FORMAT.md](evaluation/QRELS_FORMAT.md)。

### A0–A4 正式 v1 结果

2026-07-20 使用 48 条 `human_validated` qrels 完成正式运行；主要有效性结果只报告 36 条 held-out test 查询，12 条 dev 不混入主结论。每条查询重复 3 次，排名全部确定一致，Ollama 与 before-after 均未启用。

| 变体 | MRR | Recall@5 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| A0 | 0.650 | **0.621** | 0.677 | 0.622 |
| A1 | 0.508 | 0.456 | 0.570 | 0.481 |
| A2 | 0.561 | 0.547 | 0.636 | 0.548 |
| A3 | 0.512 | 0.464 | 0.626 | 0.495 |
| A4 | **0.652** | 0.514 | **0.728** | **0.648** |

A4 相对 A0 的 nDCG@10 差值仅为 +0.026，配对 bootstrap 95% CI `[-0.113, 0.171]`，不能声称完整融合总体显著优于 CLIP。A4 相对 A3 的 Metadata 增量为 +0.153，CI `[0.049, 0.275]`；全部 6 个 Metadata test 查询的 nDCG@10 从 A3 的 0.082 提升到 A4 的 1.000。Caption/Scene Graph 的固定 RRF 融合没有带来总体增益，并在部分关系查询中降低排序质量。

公开、无 UUID/查询文本的完整总体、分类、语言、延迟和置信区间见 [evaluation/retrieval_ablation_v1_summary.json](evaluation/retrieval_ablation_v1_summary.json)。私有逐查询报告保存在 Git 忽略的 `outputs/experiments/`。

## Story N0–N3 正式实验与盲评

正式 12×4 生成已完成，私有原始文本、照片路径、UUID、声明审计和 A/B 映射均位于 Git 忽略的 `evaluation/private/`。公开聚合位于 `evaluation/story_ablation_v1_summary.json`，不含故事全文或个人照片标识。

继续或校验正式 checkpoint（身份完全一致时只恢复，不重采样）：

```powershell
conda run -n torchtest python scripts/run_story_ablation.py --check-only
conda run -n torchtest python scripts/run_story_ablation.py
```

启动独立盲评页面：

```powershell
conda run -n torchtest streamlit run scripts/story_blind_review_app.py --server.port 8502
```

用户完成并锁定全部 12 个 A/B/平局选择后揭盲和聚合：

```powershell
conda run -n torchtest python scripts/analyze_story_ablation.py
```

正式结果中 N3 有 11/12 fallback；唯一直接接受的 N3 草稿又被独立声明审计发现 5 条 validator 未捕获的推断。用户已完成并锁定全部 12 组盲评，揭盲结果为 N0 胜 12、N3 胜 0、平局 0（双侧 exact binomial `p = 0.00048828125`）。当前结果支持“保守门控提高形式合规和可检测的声明安全性，但付出明显的信息量、延迟与主观可读性代价”，不支持“N3 模型文本质量全面提高”。

## Qwen3:4B Story 双轨实验

新版实验只评估冻结 digest 的本地 `qwen3:4b`，旧 Llama N0--N3 继续作为历史结果保留，不覆盖也不进入新版主结果。Faithful 轨道为 QF0--QF3，Creative 轨道为 QC0--QC2；QF2/QF3 与 QC1/QC2 分别共享完全相同的首稿，以隔离校验、修复和回退的贡献。完整预注册协议见 [evaluation/QWEN_STORY_PROTOCOL.md](evaluation/QWEN_STORY_PROTOCOL.md)。

检查模型与案例冻结清单，或按检查点运行正式生成：

```powershell
conda run -n torchtest python scripts/run_qwen_story_experiment.py --check-only
conda run -n torchtest python scripts/run_qwen_story_experiment.py --track faithful
conda run -n torchtest python scripts/run_qwen_story_experiment.py --track creative
```

两条轨道生成与声明审计已经完成。预注册目标是 24 组，但 Creative 主案例有 4/12 `error`，6 个冻结后备又有 3/6 `error`，最终只有 11 个有效 Creative 配对；系统没有事后继续挑选新案例。因此实际页面为 **23 组**（12 组 Faithful＋11 组 Creative），并明确标记实验不完整：

```powershell
conda run -n torchtest streamlit run scripts/qwen_story_blind_review_app.py --server.port 8504
```

盲评包含 12 组 QF0/QF3 与 11 组可读 QC0/QC2；全部选择和评分锁定前禁止揭盲。用户现已完成并锁定全部实际23组，之后才读取独立映射：Faithful 为 QF0 胜12、QF3胜0、平局0（双侧 exact binomial `p=0.00048828125`）；Creative 为 QC0胜11、QC2胜0、平局0（`p=0.0009765625`）。该结果只来自同一用户的个人相册案例，且 Creative 预注册目标少1组，不能外推为总体用户偏好或写成24/24完成。

正式生成和盲评全文、照片 ID 与映射均在 Git 忽略的 `evaluation/private/`，公开摘要不包含 UUID、图片路径、故事全文或 A/B 映射。

自动结果：Faithful QF3 为 7 `ok`、2 `repaired`、3 `fallback`，语言、引用和证据组覆盖均为 1.000，平均延迟 127.81 秒；QF1 为 12/12 `invalid`。Creative 主样本 QC2 为 6 `ok`、2 `repaired`、4 `error`，平均延迟 42.01 秒；QC0→QC1 重复率变化为 -0.0313，95% paired-bootstrap 区间 `[-0.0523, -0.0132]`，但 QC2 的 error 使语言和结构合规相对 QC1 各下降 0.333。盲评中 QF3 的连贯性/信息量/证据一致性分别比 QF0 低0.333/0.583/0.333；QC2 的连贯性/个性化/可信度分别比 QC0 低0.273/0.273/0.455。2,500 个声明单元的保守 agent evidence audit 与完整结果见 [evaluation/story_qwen_dual_v1_summary.json](evaluation/story_qwen_dual_v1_summary.json)。该结果不用于声称 Qwen 优于旧 Llama。

## 拍摄者心情 Metadata 单案例

Creative v6.2 可选使用人工确认的拍摄者心情；该字段不进入检索、事件分组或 Faithful Story，也不代表心率/HRV 推断。为避免小型模型忽略或错误复制标签，Qwen 负责场景叙事，应用程序把人工 Mood 确定性渲染为带 evidence ID 的第一人称拍摄者心情句，并在审计区标记为 `verified_photographer_mood`。实验仅限 `2026-07-07 · Event 1` 的 5 张热气球照片。在 Story 页面选择该 Event，切换到 Creative，为每张照片选择并保存 `neutral/calm/happy/excited/tense/sad` 之一，然后运行：

```powershell
conda run -n torchtest python scripts/run_mood_story_experiment.py
conda run -n torchtest streamlit run scripts/mood_story_blind_review_app.py --server.port 8503
conda run -n torchtest python scripts/run_mood_story_experiment.py --reveal
```

前两份故事以 `save=False` 生成，不写生产 Story 表。M0 不提供 Mood，M1 提供冻结的人工 Mood；模型、seed、temperature 和 system prompt 相同。私有全文、UUID、映射和 manifest 位于 Git 忽略的 `evaluation/private/`，公开摘要只保存聚合指标和哈希。

## 主要代码

- [app.py](app.py)：Streamlit v2 界面
- [storage.py](storage.py)：SQLite schema、迁移、FTS5/BM25 与审计记录
- [indexing_service.py](indexing_service.py)：分阶段增量 CLIP/BLIP 索引、检查点和 OOM 回退
- [vector_store.py](vector_store.py)：版本化 Chroma 图像与三元组 collection
- [retrieval_engine.py](retrieval_engine.py)：独立召回、Metadata 预过滤、RRF、图片与时间检索
- [event_organizer.py](event_organizer.py) / [event_service.py](event_service.py)：事件划分与持久化
- [story_agent.py](story_agent.py)：证据聚合、引用校验、Ollama 与确定性回退
- [story_evaluation.py](story_evaluation.py)：N0–N3 冻结定义、Story 案例与自动指标
- [qwen_story_evaluation.py](qwen_story_evaluation.py)：Qwen QF0–QF3/QC0–QC2 冻结、生成记录、自动指标、盲评与隐私摘要
- [mood_evaluation.py](mood_evaluation.py)：拍摄者 Mood M0/M1 冻结、生成记录、盲评与公开摘要
- [scene_graph_io.py](scene_graph_io.py) / [scene_graph_service.py](scene_graph_service.py)：安全 Scene Graph 往返与索引
- [evaluation.py](evaluation.py)：A0–A4 和独立时间任务评测
- [论文思路与实验跟踪.md](论文思路与实验跟踪.md)：设计决策、失败点和实验过程记录

## 已知限制

- 当前库有 1,599 个 Scene Graph 和 22,230 个三元组（376 主相册 + 1223 pic2），但关系向量的语义近邻不等于严格 S–P–O 命中；A2/A3/A4 的实际增益必须由冻结 qrels 的正式消融证明。
- A0–A4 v1 已完成，但只覆盖 2026-07-16 的 376 张个人照片和一个人工终审者；主相册此后扩充到 1599 张，但正式消融未在其上重跑，结论仍是 376 张个人相册 case study，不代表所有用户或通用图片检索。
- BLIP Caption 与 Qwen Scene Graph 是模型观察，可能漏掉小物体、人物关系、文字和细粒度动作；FTS5 的精确词项能力受 Caption 质量影响。CLIP 零样本标签已于 2026-07-27 移除（§35），不再作为检索或叙事证据。
- 中文查询会经过项目内的确定性解析/翻译规则，覆盖面不等于完整机器翻译；复杂口语可选 Ollama，但会降低严格复现性。
- v1 的 6 条中文 test 查询中，A0/A1/A3 的 nDCG@10 为 0；冻结词表没有完整翻译“热气球、鹭鸟、贪吃蛇、坐、湖边”等概念。该失败不能在查看 test 结果后原地调词并仍宣称使用同一未见测试集。
- before-after 依赖可靠且同一天的时间戳；缺少 EXIF 的照片不会仅凭文件修改时间被强行组织成可信事件。
- 事件阈值是确定性启发式规则，适合形成可解释基线，但仍需人工样本评估并可能针对个人相册调参。
- 故事引用和高风险词校验能降低无依据叙述，不能保证上游视觉模型的观察绝对正确；论文中应将其描述为 grounding/traceability，而不是事实证明。
- 正式 N0–N3 和单用户 12 组 N0/N3 盲评均已完成：N3 有 11/12 fallback，平均延迟约 273 秒、最大约 751 秒；唯一直接接受稿仍有 5 条未被词表 validator 捕获的推断。揭盲结果为 N0 胜 12、N3 胜 0、平局 0（双侧 exact binomial `p = 0.00048828125`）。这说明当前 N3 提高形式合规和可检测的事实安全性时付出了明显的信息量与可读性代价；该结果仅来自同一用户对 12 个案例的配对评价，不能外推为总体用户偏好。
- 拍摄者 Mood 的 M0/M1 单案例盲评已完成：总体偏好选择 M1；M0/M1 的连贯性为 4/3、个性化为 3/4、可信度为 3/3。该结果只说明一个热气球事件中的探索性个性化—连贯性权衡，不代表统计显著或普遍提升；M1 是 Qwen 叙事与程序注入人工确认 Mood 的混合系统，并非从照片或生理信号自动识别情绪。
- 4 GB 显存应保留 CLIP 8、BLIP 4 的保守默认值。强行增大批次可能触发 OOM；自动回退会继续运行，但会增加总耗时。
- SQLite 与 Chroma 面向单机个人相册和论文原型；大规模多用户并发、权限隔离、在线同步和分布式索引不在当前范围内。
