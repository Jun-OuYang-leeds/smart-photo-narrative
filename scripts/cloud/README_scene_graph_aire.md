# 安全 Scene Graph 离线工作流

这条链路只把 Qwen 放在 AIRE 上运行。本地导出、结果核验和数据库接入仍在
`smart-photo-narrative` 内完成。旧 JSONL 不允许按文件名或 stem 匹配个人照片。

## 1. 本地导出

先确保相册已完成 JPG 规范化，然后在 `torchtest` 环境执行：

```powershell
conda run -n torchtest python scripts/export_scene_graph_batch.py `
  --album-root photos `
  --export-dir outputs/scene_graph_batch `
  --dataset-id personal-main-v1
```

导出目录包含：

- `manifest.jsonl`：每行固定包含 `schema_version`、`dataset_id`、`photo_id`、
  `relative_path`、`sha256` 和 `file_size`。
- `images/<photo_id>.jpg`：与本地 JPG 字节完全一致，不做会改变哈希的转码。

HEIC/PNG 必须先用项目现有的 JPG 转换脚本处理。导出器遇到“扩展名是 JPG、
内容却不是 JPEG”时会直接终止。

## 2. 上传并提交 AIRE 三卡数组任务

按照项目根目录的 `aire_hpc_upload_guide.md` 上传项目代码和导出批次。首次提交前，
先创建 Slurm 日志目录；`#SBATCH --output` 指向的目录必须在调度前存在：

```bash
mkdir -p /mnt/scratch/$USER/smart-photo-narrative/logs
mkdir -p /mnt/scratch/$USER/smart-photo-narrative/outputs
```

脚本默认复用指南中已安装的 Python 环境和 Qwen 模型，也可通过 `sbatch --export`
覆盖 `PROJECT_DIR`、`SCRATCH_DIR`、`CONDA_ENV`、`MODEL_PATH`、`MANIFEST`、
`IMAGE_DIR` 或 `OUTPUT_DIR`：

```bash
sbatch scripts/cloud/run_scene_graph_array_3gpu.sbatch
```

数组任务为 `0-2%3`，每个任务只申请一张 GPU，并分别写入：

```text
qwen_scene_graph_shard_0.jsonl
qwen_scene_graph_shard_1.jsonl
qwen_scene_graph_shard_2.jsonl
```

采用指南中验证过的 Qwen2.5-VL-7B 配置：`max-model-len=1536`、
`max_tokens=640`、最多 15 个 triples。重提任务时，checkpoint 只跳过身份和
生成版本完全相同的成功非空结果；失败、空结果和坏行会继续重试。

脚本只连接计算节点上的本地 vLLM，不需要把真实云服务密钥写入项目，也不会打印
`OPENAI_API_KEY`。

## 3. 下载结果并先 dry-run

三个 shard 不需要手工拼接：

```powershell
conda run -n torchtest python scripts/import_scene_graph_results.py `
  --manifest outputs/scene_graph_batch/manifest.jsonl `
  --album-root photos `
  --results "outputs/qwen_scene_graph_shard_*.jsonl"
```

报告至少给出 `matched`、`unmatched`、`checksum_conflict`、`failed`、`empty`、
`duplicate` 和 `indexable`。只有以下条件同时成立的记录进入 `indexable_records`：

1. `dataset_id` 与 manifest 一致；
2. `photo_id` 存在于 manifest；
3. 结果 SHA-256 与 manifest 一致；
4. 本地相册仍存在完全相同 SHA-256 的图片；
5. 来源为 `remote` 或 `remote_repaired`，状态为 `success`，triples 非空。

改名不影响匹配，因为导入器会用完整内容哈希找到同一张图；basename 和 stem 从不
参与接受决策。

确认报告后，显式写出供数据库索引的干净 JSONL：

```powershell
conda run -n torchtest python scripts/import_scene_graph_results.py `
  --manifest outputs/scene_graph_batch/manifest.jsonl `
  --album-root photos `
  --results "outputs/qwen_scene_graph_shard_*.jsonl" `
  --apply `
  --output-jsonl outputs/scene_graph_indexable.jsonl `
  --report-json outputs/scene_graph_import_report.json
```

## 4. 旧 Geograph JSONL

旧文件没有 `photo_id` 和原始 SHA-256，因此只能在提供原始 Geograph 图片根目录后
做现场哈希：

```powershell
conda run -n torchtest python scripts/import_scene_graph_results.py `
  --manifest path/to/geograph_manifest.jsonl `
  --album-root path/to/geograph_images `
  --results path/to/qwen_scene_graph_outputs_full.jsonl `
  --legacy-root path/to/geograph_images
```

旧记录只有在 manifest 的 `dataset_id` 明确以 `geograph` 开头且完整哈希匹配时才
可索引。对个人相册运行只会得到审计报告，不会产生可索引记录；同名但不同内容会
被列为 `checksum_conflict`。
