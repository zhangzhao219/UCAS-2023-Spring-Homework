## 数据说明

- 条目含义

  - id：唯一标识，用于与音频模态对齐
  - transcription: 文本
  - emotion: 情绪，（音频模态采用的）映射关系为 {"neu": 0, "hap": 1, "ang": 2, "sad": 3}

- 子集划分

  按照顺序划分 train: val: test = [0.8, 0.1, 0.1]，即将 pickle 文件保序地平均分为十份，前八份为 train，第九份为 val，最后为 test （这一步目前没有做，需要一定的处理）
