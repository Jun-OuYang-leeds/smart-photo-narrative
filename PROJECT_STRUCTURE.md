# Smart Photo Narrative 项目结构

更新日期：2026-07-17

本文区分“运行必需内容、实验产物、回滚备份和历史内容”。不要仅凭目录体积判断能否删除。

## 核心运行内容

| 路径 | 用途 | 处理原则 |
|---|---|---|
| `app.py` | Streamlit 前端入口 | 保留在项目根目录 |
| 根目录 `*.py` | 存储、索引、检索、事件、故事及模型服务 | 当前使用顶层模块导入，不宜随意移入 `src/` |
| `photos/` | 个人相册原图/JPG 工作副本 | 私有数据；不要提交，不要批量移动 |
| `data/smart_photo.db` | 当前 SQLite 主数据库，含照片、Caption、Scene Graph、FTS5 等 | 运行必需；修改前备份 |
| `indexes/chroma_v2/` | 当前 CLIP 与 Scene Graph 向量索引 | 运行必需；应与 SQLite 保持一致 |
| `scripts/` | 导入、恢复、烟测和云端 Scene Graph 工具 | 保留 |
| `tests/` | 单元测试和 Streamlit AppTest | 保留 |
| `evaluation/` | A0–A4 检索评估代码、qrels 格式与报告目录 | 保留 |

## 已整理的实验产物

```text
outputs/
├─ captions/                  # BLIP/BLIP2 比较结果
└─ scene_graph/
   ├─ raw/                    # AIRE/Qwen 原始返回与压缩包
   ├─ recovered/              # 旧批次恢复后的可导入 JSONL 和审计报告
   ├─ reports/                # Scene Graph 搜索烟测报告
   └─ backups/                # 正式导入前的 SQLite/Chroma 回滚副本
```

`outputs/` 含个人照片派生信息并已被 Git 忽略。整理只改变了文件路径，没有改变文件内容；README 和实验跟踪文档中的对应路径已同步更新。

## 回滚与历史内容

| 路径 | 含义 | 建议 |
|---|---|---|
| `photo_conversion_20260716_jpg_normalization/` | HEIC/PNG 等统一转 JPG 时留下的原文件与清单 | 在确认全部照片、EXIF 和数据库无误前保留 |
| `chroma_db/` | 旧版向量索引路径，仍有 Git 历史状态 | 不作为当前 `chroma_v2` 使用；暂不删除 |
| `chroma_db_legacy_20260716_pre_multimodal/` | 多模态升级前冻结的旧 Chroma 副本 | 仅用于回滚/审计，已被 Git 忽略 |
| `docs/legacy/test_output_ollama_20260222.txt` | 早期 Ollama 测试输出 | 历史记录，不参与运行 |
| `Msc/` | 指向独立 GitHub 仓库的嵌套 Git 元数据，目前不是主项目代码 | 不参与运行；因 Windows/工作区保护未强制移动或删除 |

## 可再生成内容

- `cache/`：运行缓存；程序按需重建。
- `__pycache__/`、`*.pyc`：Python 字节码缓存；本次已清理，运行后可能再次出现。
- `evaluation/reports/`：评估报告；应由固定配置和 qrels 重新生成。

## 根目录为什么仍保留多个 Python 文件

当前模块之间普遍使用 `from storage import ...`、`from config import ...` 这类顶层导入。把它们机械移动到 `src/` 会同时影响 Streamlit、脚本、测试和 Conda 启动方式。因此本次仅整理生成物与历史输出，没有进行高风险的包结构重构。
