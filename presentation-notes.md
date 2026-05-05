# 内容过滤部分 Pre 文档

本文档用于小组作业展示时讲解内容过滤 Content-Based Filtering 部分。

## 1. 本部分负责什么

本部分实现的是推荐系统中的内容过滤模块。

目标是：

- 输入一个 `userId`
- 根据该用户历史高评分电影的内容特征构建兴趣画像
- 推荐该用户还没有看过、但内容上相似的电影

最终输出：

- `movieId`
- `title`
- `genres`
- `raw_similarity`
- `content_score`

其中 `content_score` 是归一化到 `0-1` 的内容过滤推荐分数，后续可以和协同过滤结果进行加权融合。

## 2. 内容过滤是什么

内容过滤的核心思想是：

如果用户过去喜欢某些内容特征的物品，那么系统就继续推荐内容特征相似的物品。

在电影推荐场景中，物品就是电影，内容特征可以来自：

- 电影类型 `genres`
- 电影标签 `tags`
- 电影标题 `title`
- 电影简介 `overview`
- 导演、演员、年份等信息

本项目主要使用 MovieLens 数据集中的 `genres + tags` 作为电影内容特征。

## 3. 和协同过滤的区别

协同过滤 Collaborative Filtering 更关注：

- 用户和用户之间是否相似
- 电影和电影之间是否被相似用户喜欢

内容过滤 Content-Based Filtering 更关注：

- 当前用户喜欢过什么内容
- 候选电影本身是否和这些内容相似

简单对比：

- 协同过滤：和你相似的人喜欢什么，我就推荐什么
- 内容过滤：你过去喜欢什么类型，我就推荐类似内容

## 4. 为什么需要内容过滤

内容过滤的优点：

1. 不完全依赖其他用户的数据。
2. 推荐结果具有一定可解释性。
3. 对新电影更友好，只要新电影有内容特征，就可以参与推荐。
4. 可以和协同过滤互补，用于混合推荐。

内容过滤的局限：

1. 容易推荐和用户历史兴趣过于相似的内容，探索性较弱。
2. 如果电影内容特征太少，推荐质量会受到限制。
3. 对完全没有历史行为的新用户仍然不够友好。

因此在最终系统中，内容过滤更适合和协同过滤组合使用。

## 5. 本项目使用的数据

本项目使用 MovieLens `ml-latest-small` 数据集。

核心文件：

- `movies.csv`
- `ratings.csv`
- `tags.csv`

字段说明：

- `movies.csv`: `movieId`, `title`, `genres`
- `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`
- `tags.csv`: `userId`, `movieId`, `tag`, `timestamp`

示例：

```text
title: Toy Story (1995)
genres: Adventure|Animation|Children|Comedy|Fantasy
tags: pixar, fun, animation
```

在本项目中，我们不使用电影海报、剧情简介、演员、导演等额外信息，只使用 `genres` 和 `tags` 中的分类/标签词来表示电影内容。

以上面的 Toy Story 为例，它的内容特征会被整理成：

```text
Adventure Animation Children Comedy Fantasy pixar fun animation
```

也就是说，模型后续只会考虑候选电影和用户兴趣在这些分类/标签词上的相似程度。

需要注意的是，这一串词只是 Toy Story 这部电影的示例。

对于数据集中的其他电影，也会用各自的 `genres + tags` 生成对应的内容词。

例如，动作片可能包含 `Action Thriller Crime`，爱情片可能包含 `Romance Drama`。

然后模型会把每部电影整理后的内容词做 TF-IDF 文本向量化。

## 6. 整体实现流程

本模块的流程如下：

1. 加载 movies.csv、ratings.csv、tags.csv
2. 处理每部电影的 genres 和 tags
3. 使用 TF-IDF 将电影内容转成向量
4. 找到指定用户的历史评分
5. 取评分 >= 4.0 的电影作为用户喜欢的电影
6. 根据这些高评分电影构建用户兴趣画像
7. 计算用户画像和所有电影向量的余弦相似度
8. 去掉用户已经评分过的电影
9. 将分数归一化为 content_score
10. 按 content_score 从高到低返回推荐结果

## 7. TF-IDF 的作用

TF-IDF 是一种常见的文本特征表示方法，用于把文本转成数值向量。

在本项目中，每部电影的 `genres + tags` 可以看成一段文本。TF-IDF 会衡量一个词对某部电影的重要程度。

直观理解：

如果一个词在某部电影中出现，但在所有电影中并不常见，它就更能代表这部电影。

例如：

- `Adventure`、`Comedy` 这类词可能很常见
- `pixar`、`superhero`、`space` 这类标签则更有区分度

因此 TF-IDF 比简单地统计词频更适合表示电影内容。

### 候选电影内容向量如何构建

候选电影内容向量不是手工指定的，而是由 TF-IDF 自动计算出来的。

具体来说，每部电影都会先被整理成一段“内容文本”：

`genres_text + tag_text`

例如：

```text
Adventure Animation Children Comedy Fantasy pixar fun animation
```

然后 `TfidfVectorizer` 会扫描所有电影的内容文本，建立一个词表。词表中可能包含：

`Action`, `Adventure`, `Comedy`, `Fantasy`, `pixar`, `superhero`, `space` 等词。

对于每一部电影，TF-IDF 会计算这部电影在词表中每个词上的权重。这样一部电影就会变成一个向量：

```text
movie_vector = [Action权重, Adventure权重, Comedy权重, Fantasy权重, pixar权重, ...]
```

如果某部电影包含 `Action` 和 `Adventure`，那么这两个位置的权重会比较高；如果不包含 `Romance`，那么 `Romance` 对应位置就是 0。

在代码中，对应的是：

```python
movie_matrix = vectorizer.fit_transform(movies_content["content"])
```

这里的 `movie_matrix` 就是所有电影的内容向量矩阵：

- 每一行：一部电影的内容向量
- 每一列：一个 `genres` 或 `tags` 中出现过的词
- 每个数值：该词对这部电影的重要程度

所谓“候选电影内容向量”，就是某部候选电影在 `movie_matrix` 中对应的那一行。

需要注意的是，所有电影都会先生成内容向量。后面做推荐时，只是把用户已经评分过的电影过滤掉，剩下的电影才作为候选电影参与排序。

## 8. 用户兴趣画像如何构建

对于某个用户，比如 `userId = 1`，先从 `ratings.csv` 中找到他的历史评分：

```python
user_ratings = ratings[ratings["userId"] == 1]
```

然后选择评分较高的电影：

```python
liked = user_ratings[user_ratings["rating"] >= 4.0]
```

这些电影被认为代表用户偏好。

如果用户喜欢过多部电影，就把这些电影的 TF-IDF 向量按照评分加权平均，得到用户兴趣画像。

可以理解为：

用户兴趣画像 = 用户喜欢过的电影内容向量的加权平均

简化公式：

```text
user_profile = sum(rating_i * movie_vector_i) / sum(rating_i)
```

评分越高的电影，对用户画像影响越大。

## 9. 如何计算推荐分数

得到用户兴趣画像后，需要判断“某部候选电影”和“这个用户的兴趣”有多接近。

这里使用的是余弦相似度 cosine similarity。

可以先用一个直观例子理解：

- 用户兴趣画像：`Action`, `Adventure`, `Fantasy`, `Comedy` 的权重比较高
- 候选电影 A：`Action`, `Adventure`, `Fantasy` 的权重也比较高
- 候选电影 B：`Romance`, `Documentary` 的权重比较高

那么候选电影 A 和用户兴趣的方向更接近，所以余弦相似度更高；候选电影 B 的内容方向和用户兴趣差得比较远，所以相似度更低。

这里的“方向”可以理解为：

这部电影的内容重点，是否和用户过去喜欢的内容重点一致。

数学上，余弦相似度衡量的是两个向量夹角的大小：

```text
cosine_similarity(A, B) = A dot B / (|A| * |B|)
```

在本项目中：

- `A` = 用户兴趣向量
- `B` = 候选电影内容向量

所以它不是在计算电影和用户之间的因果关系，而是在计算：

电影内容特征和用户兴趣特征之间的相似程度。

可以把它理解为一个“内容匹配分数”：

电影内容越接近用户兴趣画像，相似度越高，也就越应该被推荐。

代码中会先得到 `raw_similarity`，然后把它归一化到 `0-1`，得到 `content_score`。

## 10. 为什么要过滤已看过电影

推荐系统的目标是推荐用户可能感兴趣的新电影。

所以在生成推荐结果时，需要去掉用户已经评分过的电影：

用户评分过的电影，可以理解为用户已经看过或已经表达过态度的电影。

本模块默认只推荐用户没有评分过的电影。

## 11. 内容过滤部分的输出形式

示例输出：

```text
movieId,title,genres,raw_similarity,content_score
117646,Dragonheart 2: A New Beginning (2000),Action|Adventure|Comedy|Drama|Fantasy|Thriller,0.772052,1.0
```

含义：

- `movieId`: MovieLens 中的电影编号
- `title`: 推荐电影标题
- `genres`: 电影类型
- `raw_similarity`: 原始余弦相似度
- `content_score`: 归一化后的内容过滤推荐分数

如果 `content_score = 1.0`，说明它在当前候选电影中和用户兴趣画像最相似。

## 12. 代码对应关系

核心文件：

`content_based.py`

主要类：

```python
ContentBasedRecommender
```

核心方法：

```python
fit(movies, ratings, tags)
```

作用：

构建电影内容文本，并使用 TF-IDF 训练电影内容向量。

```python
recommend(user_id=1, top_n=10)
```

作用：

返回某个用户的 Top-N 内容过滤推荐结果。

```python
score_movies(user_id=1)
```

作用：

返回某个用户所有未看过电影的内容过滤分数。这个方法主要给后续混合推荐模块使用。

## 13. 和混合推荐如何整合

本模块输出：

`userId`, `movieId`, `content_score`

协同过滤模块建议输出：

`userId`, `movieId`, `collaborative_score`

整合时按 `userId + movieId` 合并，然后加权：

```python
final_score = 0.7 * collaborative_score_norm + 0.3 * content_score
```

注意：

- `content_score` 已经是 `0-1`
- 协同过滤分数如果是 `0.5-5.0`，需要先归一化到 `0-1`

这样可以避免某一部分因为数值范围更大而主导最终结果。

## 14. 内容过滤部分总结

本部分实现了基于内容的电影推荐。

我们使用 MovieLens 中的 `genres` 和 `tags` 构造电影内容特征，并通过 TF-IDF 将电影文本转成向量。

然后根据用户历史高评分电影构建用户兴趣画像，计算该画像与未看过电影之间的余弦相似度，得到内容过滤推荐分数 `content_score`。

该分数已经归一化到 `0-1`，可以直接用于后续和协同过滤结果进行加权融合。

## 15. 可能被问到的问题

Q1: 为什么选择评分 `>= 4.0` 的电影作为用户喜欢的电影？

A: MovieLens 评分范围通常是 `0.5-5.0`，`4.0` 及以上可以认为用户比较喜欢。这样可以减少低评分电影对用户兴趣画像的干扰。

Q2: 如果某个用户没有评分 `>= 4.0` 的电影怎么办？

A: 代码中设置了 fallback 机制，会选择该用户评分最高的前几部电影来构建画像，避免无法推荐。

Q3: 为什么使用 TF-IDF，而不是直接用 genre one-hot？

A: TF-IDF 可以同时处理 `genres` 和 `tags`，并且能够降低过于常见词的权重，提高更有区分度标签的影响。

Q4: 内容过滤是否使用了其他用户的评分？

A: 不直接使用其他用户的偏好。它主要根据当前用户自己的历史评分和电影内容特征进行推荐。

Q5: 为什么还需要协同过滤？

A: 内容过滤容易局限在用户过去喜欢的内容范围内，探索性不足。协同过滤可以从相似用户的行为中发现用户可能感兴趣但内容特征不同的电影，两者结合效果更稳。
