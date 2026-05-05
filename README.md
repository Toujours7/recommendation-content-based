# MovieLens 内容过滤推荐模块

本目录实现的是推荐系统作业中的 **内容过滤 Content-Based Filtering** 部分。该模块可以独立运行 demo，也可以给后续的协同过滤 + 加权融合模块提供标准化的 `content_score`。

## 功能说明

内容过滤模块的核心目标是：

```text
输入 userId
输出该用户未看过电影的内容推荐分数 content_score
```

实现流程：

1. 从 `ratings.csv` 中找到指定用户的历史评分记录。
2. 选择该用户评分较高的电影，默认 `rating >= 4.0`。
3. 使用电影的 `genres` 和可选的 `tags` 构造电影内容文本。
4. 使用 TF-IDF 将电影内容文本向量化。
5. 将用户喜欢过的电影向量按评分加权平均，得到用户兴趣画像。
6. 计算用户兴趣画像与所有未看过电影之间的余弦相似度。
7. 将相似度归一化到 `0-1`，得到 `content_score`。

## 项目文件

```text
content_based.py       内容过滤核心代码
run_content_demo.py    命令行运行 demo
server_content_demo.py 模拟服务端 HTTP 接口
requirements.txt       Python 依赖
outputs/               推荐结果导出目录
```

## 数据准备

使用 MovieLens `ml-latest-small` 数据集。下载后目录结构应为：

```text
data/ml-latest-small/
  movies.csv
  ratings.csv
  tags.csv
  links.csv
```

本模块必须使用：

```text
movies.csv
ratings.csv
```

可选使用：

```text
tags.csv
```

其中：

```text
movies.csv: movieId, title, genres
ratings.csv: userId, movieId, rating, timestamp
tags.csv: userId, movieId, tag, timestamp
```

## 安装依赖

如果已经有虚拟环境 `.venv`，可以直接使用项目内 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果没有虚拟环境，可以先创建：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果下载速度较慢，可以使用清华源：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行 demo

推荐 `userId = 1` 的 Top 10 电影：

```powershell
.\.venv\Scripts\python.exe run_content_demo.py --data-dir data/ml-latest-small --user-id 1 --top-n 10
```

导出推荐结果：

```powershell
.\.venv\Scripts\python.exe run_content_demo.py --data-dir data/ml-latest-small --user-id 1 --top-n 10 --output outputs/user_1_content.csv
```

只使用 `genres`，不使用 `tags.csv`：

```powershell
.\.venv\Scripts\python.exe run_content_demo.py --data-dir data/ml-latest-small --user-id 1 --top-n 10 --no-tags
```

## 模拟服务端调用

如果想模拟后端服务调用内容过滤模块，可以运行：

```powershell
.\.venv\Scripts\python.exe server_content_demo.py --data-dir data/ml-latest-small --host 127.0.0.1 --port 8000
```

启动后访问：

```text
http://127.0.0.1:8000/recommend/content?user_id=1&top_n=10
```

也可以用 PowerShell 请求接口：

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/recommend/content?user_id=1&top_n=10"
```

健康检查接口：

```text
http://127.0.0.1:8000/health
```

接口返回 JSON，结构类似：

```json
{
  "userId": 1,
  "topN": 10,
  "count": 10,
  "recommendations": [
    {
      "movieId": 117646,
      "title": "Dragonheart 2: A New Beginning (2000)",
      "genres": "Action|Adventure|Comedy|Drama|Fantasy|Thriller",
      "raw_similarity": 0.772052,
      "content_score": 1.0
    }
  ]
}
```

## 输出字段

demo 导出的 CSV 示例字段：

```text
movieId,title,genres,raw_similarity,content_score
```

字段含义：

```text
movieId          MovieLens 中的电影 ID
title            电影标题
genres           电影类型
raw_similarity   用户兴趣画像与电影内容向量的原始余弦相似度
content_score    归一化后的内容推荐分数，范围 0-1
```

`content_score` 越接近 `1`，表示该电影和该用户的历史兴趣越相似。

## 给整合模块的调用方式

整合时建议直接调用 `content_based.py`，不要只读取 Top 10 的 demo 文件。Top 10 文件主要用于展示，融合时最好使用该用户所有未看过电影的完整分数表。

```python
from content_based import ContentBasedRecommender, load_movielens

movies, ratings, tags = load_movielens("data/ml-latest-small")

recommender = ContentBasedRecommender(use_tags=True)
recommender.fit(movies, ratings, tags)

content_scores = recommender.score_movies(user_id=1)
content_scores.insert(0, "userId", 1)

content_scores = content_scores[
    ["userId", "movieId", "title", "genres", "content_score"]
]
```

整合模块可以使用：

```text
userId, movieId, content_score
```

作为内容过滤部分的输入结果。

## 与协同过滤加权融合

假设协同过滤模块输出：

```text
userId, movieId, collaborative_score
```

内容过滤模块输出：

```text
userId, movieId, content_score
```

可以按 `userId + movieId` 合并，然后加权：

```python
final_score = 0.7 * collaborative_score_norm + 0.3 * content_score
```

注意：本模块输出的 `content_score` 已经归一化到 `0-1`。如果协同过滤输出的是预测评分，例如 `0.5-5.0`，需要先归一化成 `0-1`，再参与融合。

## 可调参数

```python
ContentBasedRecommender(
    min_like_rating=4.0,
    fallback_profile_size=5,
    use_tags=True,
    stop_words="english",
)
```

参数说明：

```text
min_like_rating       用户评分达到多少分才认为是喜欢，默认 4.0
fallback_profile_size 如果用户没有评分 >= 4.0 的电影，则取评分最高的前 N 部构建画像
use_tags              是否使用 tags.csv 中的标签信息
stop_words            TF-IDF 的停用词设置
```

## 说明

本模块是内容过滤推荐，不使用其他用户的行为数据来判断相似用户。它只根据当前用户自己评分较高的电影内容特征，寻找内容上相似的未看过电影。
