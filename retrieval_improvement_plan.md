# smart-photo-narrative 检索能力提升建议

本文基于 `smart-photo-narrative` 与 `Simulate SnapSeek3.0` 的代码结构和检索实现进行对比，重点分析当前项目在检索方面的短板，并给出可执行的改进路线。本文只讨论方案，不包含代码修改。

## 1. 两个项目的检索定位差异

`smart-photo-narrative` 更像一个本地智能相册应用。它的核心流程是扫描 `photos/` 目录，提取 EXIF 元数据，使用 CLIP 生成图片向量，使用 BLIP 生成 caption，然后写入 ChromaDB。检索功能主要服务于相册浏览和照片查找。

`Simulate SnapSeek3.0` 更像一个专门面向 lifelog retrieval 的检索原型。它不仅使用图文向量，还额外引入 OCR、scene graph、ADL 活动标签、TF-IDF 稀疏文本、时间关系查询和用户反馈重排。因此它的检索系统更偏“多信号融合”和“交互式检索”。

简言之：

- `smart-photo-narrative` 当前是“图像语义搜索 + caption 关键词搜索”。
- `Simulate SnapSeek3.0` 是“多模态、多字段、多阶段排序系统”。

## 2. smart-photo-narrative 当前检索机制

当前检索主要位于 `hybrid_retriever.py`。

已有三种模式：

- `vector`：把用户查询编码为 CLIP 文本向量，在 ChromaDB 中检索相似图片。
- `keyword`：在 BLIP caption 文本中做关键词包含匹配。
- `hybrid`：先使用 CLIP 向量检索候选，再根据 caption 中命中的查询词数量重新加分。

当前 hybrid 评分大致是：

```text
final_score = vector_weight * clip_score + (1 - vector_weight) * keyword_score
```

这个思路是对的，但候选生成不够稳。它主要依赖向量检索先召回一批候选。如果某张图片 caption 精确命中查询词，但 CLIP 相似度较低，没有进入向量候选池，那么 hybrid 阶段也不会看到它。

## 3. 对比 SnapSeek 后的主要提升点

### 3.1 候选召回应从单一路径改成多路径并集

当前问题：

hybrid 检索通常先取 `top_k * 3` 个 CLIP 向量结果，然后只在这些结果里计算 caption keyword 分数。这会导致强关键词命中的图片被漏掉。

建议改成多路召回：

```text
candidate_pool =
    topN vector results
  ∪ topN caption/BM25 results
  ∪ topN tag results
  ∪ topN metadata results
```

也就是说，先从多个检索通道分别找候选，再把候选合并成一个去重集合，然后统一重排。

这些通道可以这样理解：

- `topN vector`：适合找视觉语义相似的图片，例如 “sunset beach”“person with laptop”。
- `topN caption/BM25`：适合找 caption 精确描述过的内容，例如 “red apple”“bowl of soup”。
- `topN tag`：适合粗粒度类别匹配，例如 `food`、`document`、`city`、`cat`。
- `topN metadata`：适合时间、地点、文件名、尺寸、EXIF 等结构化条件。

这样做的好处是召回更稳：只要某张图片在任意一个信号上很强，它就有机会进入最终排序。

### 3.2 用 RRF 融合多个排名

RRF 是 Reciprocal Rank Fusion，中文可以理解为“倒数排名融合”。它不要求不同检索通道的分数可比，只看每个候选在各个通道里的排名。

公式：

```text
rrf_score(doc) = Σ 1 / (k + rank_i(doc))
```

其中：

- `rank_i(doc)` 是某张图片在第 `i` 个检索通道中的排名。
- `k` 是平滑常数，常用 `60`。
- 如果某张图片没有出现在某个通道中，就不加该通道分数。

例子：

```text
查询：apple on table

vector 排名：
1. apple.png
2. Cambridge_food.jpg
3. IMG_2685.JPG

caption 排名：
1. apple.png
2. gradient.png
3. finger.jpg

tag 排名：
1. apple.png
2. Cambridge_food.jpg

apple.png 在多个通道都靠前，RRF 分数会最高。
```

RRF 的优点：

- 不需要强行比较 CLIP 分数和 BM25 分数。
- 对异常高分不敏感。
- 多个通道都认为相关的图片会自然排到前面。
- 实现简单，适合作为第一版融合方案。

推荐优先级：高。

### 3.3 或者使用归一化加权融合

另一种方式是把不同通道的原始分数归一化到 `0-1`，再加权求和。

示例：

```text
final_score =
    0.50 * vector_score_norm
  + 0.25 * caption_score_norm
  + 0.15 * tag_score_norm
  + 0.10 * metadata_score_norm
```

优点：

- 分数解释更直观。
- 可以手动调权重。
- 适合后续做 UI 参数控制，例如用户调节 “语义权重 / 关键词权重”。

缺点：

- 不同通道分数分布可能差异很大。
- 归一化方式会影响最终排名。
- 比 RRF 更需要调参。

建议：

- 第一版先用 RRF。
- 如果后续希望更可控，再做归一化加权融合。

### 3.4 caption 关键词检索应升级为 BM25 或 TF-IDF

当前 keyword 检索较弱，主要问题是：

- 多词查询处理不充分。
- 只做简单包含匹配，不能处理词频、逆文档频率、短语权重。
- 不擅长区分 “apple” 这种强关键词和 “a / the / on” 这种弱词。

建议增加一个 caption 文本索引：

```text
caption_text = BLIP caption + tags + location + file name
```

然后用以下任一方式检索：

- `TfidfVectorizer`：实现简单，依赖少。
- BM25：文本检索更经典，关键词排序更稳。

如果暂时不引入新库，可以先用 `sklearn.feature_extraction.text.TfidfVectorizer`。

### 3.5 标签检索应从 JSON 字符串包含改为规范化标签匹配

当前标签以 JSON 字符串存入 ChromaDB，例如：

```json
["food", "Western food", "dessert"]
```

然后通过 `$contains` 做字符串匹配。这种方式简单，但不够稳：

- 容易出现子串误匹配。
- 不利于组合查询。
- 不利于统计标签权重。

建议：

- 保留原有 JSON tags 作为展示字段。
- 额外建立 tag inverted index：`tag -> photo_ids`。
- 检索时通过标签倒排表快速召回候选。

标签分数可以这样设计：

```text
tag_score = matched_tag_count / query_tag_count
```

或者：

```text
tag_score = max(CLIP tag confidence for matched tags)
```

### 3.6 元数据过滤应前置，而不是检索后过滤

当前日期范围过滤是在 Python 中后处理。问题是如果向量检索先取 20 个，日期过滤后可能只剩很少结果，甚至为空，但数据库中其实有符合日期的照片。

建议索引时增加数值字段：

```text
timestamp_int: 20250813171351
date_yyyymmdd: 20250813
year: 2025
month: 202508
```

这样可以先按元数据过滤候选，再做向量或文本排序。

理想流程：

```text
1. 根据日期、地点、标签等过滤出 eligible_ids
2. vector / caption / tag / metadata 各通道只在 eligible_ids 中检索
3. 合并候选
4. 融合排序
```

### 3.7 中文检索需要修复或重做

当前项目中有中文到英文查询映射，但源码里映射内容出现乱码。由于 CLIP 和 BLIP 使用英文效果更稳定，中文查询如果不翻译，会影响召回。

建议分三步：

1. 修复乱码映射表，至少保证常见词可用。
2. 增加 query rewrite，例如把 “海边日落” 转成 “sunset beach ocean sea golden hour”。
3. 后续可考虑多语言模型或本地翻译模型。

短期最划算的做法是维护一份干净的中文词典：

```text
海边 -> beach ocean sea
日落 -> sunset golden hour orange sky
猫 -> cat kitten feline
食物 -> food meal cuisine dish
截图 -> screenshot screen capture
```

### 3.8 增加 OCR 和 scene graph 字段

SnapSeek 的一个明显优势是 scene graph 和 OCR。

`smart-photo-narrative` 当前只有 BLIP caption。caption 通常是一句话，适合粗略描述，但对关系型查询不够强，例如：

```text
person holding phone beside keyboard
laptop on desk
cash payment in cafe
text on screenshot
```

建议新增两个字段：

```text
ocr_text: 图片中文字
scene_graph_text: 主体-关系-客体文本
```

scene graph 可以先做轻量版，不一定一开始就接大模型：

```text
caption: a person using a laptop at a desk
scene_graph_text:
  person use laptop
  laptop be on desk
  person sit near desk
```

然后将它们纳入候选召回：

```text
topN scene_graph/BM25
topN ocr/BM25
```

### 3.9 暴露已有的以图搜图能力

`hybrid_retriever.py` 中已经有 `search_by_image()`，但当前 Streamlit 搜索页没有入口。

建议在 UI 中增加：

- 上传一张图片搜索相似照片。
- 或选择当前相册中的一张图片，点击 “Find similar”。

这是一个低成本、高收益的检索功能。

### 3.10 增加用户反馈重排

SnapSeek 支持用户点击 Relevant / Irrelevant，然后重新排序。`smart-photo-narrative` 可以做一个简化版。

思路：

```text
用户标记相关图片：
  提升这些图片及其向量邻居

用户标记不相关图片：
  降低这些图片及其向量邻居
```

可以使用 Rocchio 风格更新查询向量：

```text
new_query =
    original_query
  + alpha * mean(relevant_embeddings)
  - beta * mean(irrelevant_embeddings)
```

这对个人相册很有用，因为同一个人的搜索意图经常很主观。

## 4. 推荐的新检索流程

建议将检索拆成四个阶段。

### 阶段一：解析查询和过滤条件

输入：

```text
query = "apple on table"
filters = {
  start_date,
  end_date,
  tags,
  location_keyword,
  caption_keyword
}
```

处理：

- 如果是中文，先翻译或改写成英文。
- 提取可能的标签词。
- 提取日期、地点、文件名等元数据条件。

### 阶段二：多通道候选召回

建议取较大的候选数，例如 `candidate_k = max(top_k * 5, 50)`。

```text
vector_candidates = CLIP text-to-image topN
caption_candidates = caption BM25/TF-IDF topN
tag_candidates = tag match topN
metadata_candidates = metadata match topN
```

然后合并：

```text
candidate_ids = union(
  vector_candidates,
  caption_candidates,
  tag_candidates,
  metadata_candidates
)
```

### 阶段三：融合排序

推荐先用 RRF：

```text
final_score[id] =
    rrf(vector_rank[id])
  + rrf(caption_rank[id])
  + rrf(tag_rank[id])
  + rrf(metadata_rank[id])
```

可选加入权重：

```text
final_score[id] =
    0.50 * rrf(vector_rank[id])
  + 0.25 * rrf(caption_rank[id])
  + 0.15 * rrf(tag_rank[id])
  + 0.10 * rrf(metadata_rank[id])
```

### 阶段四：返回结果和解释

每个结果最好附带：

```text
score
matched_modalities: ["vector", "caption", "tag"]
caption
tags
datetime
location
```

这样用户能知道为什么这张图片被召回。

## 5. RRF 融合示例

假设查询：

```text
food in Cambridge
```

各通道返回：

```text
vector:
1. Cambridge_food.jpg
2. IMG_4013.JPG
3. IMG_4021.JPG

caption:
1. Cambridge_food.jpg
2. IMG_4373.JPG
3. IMG_4013.JPG

tag:
1. Cambridge_food.jpg
2. IMG_4373.JPG

metadata:
1. Cambridge_food.jpg
```

RRF 会让 `Cambridge_food.jpg` 排名最高，因为它在多个通道都靠前。`IMG_4013.JPG` 虽然 vector 排第二，但 caption 也出现了，因此比只在单一通道出现的图片更可靠。

## 6. 实施优先级

### P0：立即值得做

1. 修复中文查询映射乱码。
2. 修改 hybrid 候选生成：从单一路径改成多路径并集。
3. caption keyword 从简单 `$contains` 升级为 TF-IDF 或 BM25。
4. 日期过滤增加数值字段，避免检索后过滤导致结果不足。

### P1：体验提升明显

1. 在 UI 中暴露以图搜图功能。
2. 返回 `matched_modalities`，解释结果来自 vector、caption、tag 还是 metadata。
3. 标签检索改为倒排索引或规范化字段。
4. 增加文件名、路径、caption、tags 的统一文本检索字段。

### P2：更接近 SnapSeek 的能力

1. 增加 OCR。
2. 增加 scene graph 或结构化视觉描述。
3. 增加时序查询：before、after、within。
4. 增加用户反馈重排。

## 7. 推荐的第一版改造目标

如果只做一轮检索提升，建议目标定为：

```text
多路召回 + RRF 融合 + 更好的 caption 文本索引
```

第一版不需要引入复杂架构，也不必完全照搬 SnapSeek。只要把候选召回从：

```text
vector only -> rerank by caption
```

改为：

```text
vector + caption + tag + metadata -> union -> RRF rerank
```

检索稳定性就会明显提升。

## 8. 最终建议

`smart-photo-narrative` 的优势是结构简单、适合本地相册、已经有 CLIP/BLIP/ChromaDB 基础。它不需要变成完整的 SnapSeek，但可以吸收 SnapSeek 的几个核心思想：

- 多信号召回，而不是只依赖向量。
- 多字段检索，而不是只搜 caption。
- 多阶段排序，而不是一次 ChromaDB 查询结束。
- 结果可解释，让用户知道为什么命中。
- 后续增加反馈和时间关系，让个人相册搜索更贴近真实使用。

最推荐的路线是先把 hybrid 检索重做成“候选并集 + RRF 融合”。这是投入最小、收益最大的提升点。
