# 自然语言处理 课程实验一：信源熵计算与齐夫定律验证

本项目是自然语言处理课程的**实验一（课程作业-1-A）**，旨在通过实际编程操作，加深对自然语言统计特性的理解。

本实验包含完整的流程代码：**从网络爬虫收集双语预料、数据清洗、概率与信息熵计算，到齐夫定律(Zipf's Law)的可视化验证与结果分析**。

---

## 📂 代码框架与目录结构

项目的核心逻辑在 `pj1` 文件夹下，包含以下主要模块：

```text
pj1/
├── data/                  # 存放所有爬取和清洗后的中英文语料数据 (json 格式)
│   ├── bbc_data_*.json    # 收集的英文数据
│   └── sina_data_*.json   # 收集的中文数据
├── quotes_spider/         # Scrapy 爬虫项目目录
│   ├── spiders/           # 包含具体的爬虫脚本 (en_corpus.py, zh_corpus.py)
│   ├── pipelines.py       # 关键！包含文本清洗（去乱码/HTML）的核心逻辑
│   └── run_spider.sh      # 运行爬虫的一键脚本
├── statistics/            # 统计计算与信息熵分析
│   ├── calc_entropy.py    # 核心统计脚本 (词频统计、分词、计算信息熵)
│   ├── run_calc.sh        # 计算不同规模下数据的批处理自动化脚本
│   └── results/           # 存放运行后产生的各类词频分布及日志结果 
├── verification/          # 用于验证理论规律
│   ├── verify_zipf.py     # 计算并绘制对数图以验证齐夫定律
│   └── results/           # 存放生成的可视化图表
├── report/                # LaTeX 实验报告源码及对应图片
│   └── report.tex      
├── run_experiments.sh     # 串联所有实验步骤(爬取->统计->验证)的顶层运行脚本
├── README.md              # 项目说明文档
├── scrapy.cfg             # Scrapy 框架配置文件
└── requirements.txt       # 项目依赖库列表
```

---

## 🛠️ 环境依赖与安装

本项目依赖的库包括网络爬取、中文处理和科学计算，需要提前安装：

```bash
# 推荐使用虚拟环境 (Virtual Environment)
pip install -r requirements.txt # 如果有
# 或者手动安装以下核心库:
pip install Scrapy jieba matplotlib
```

---

## 🚀 使用方法

您可以选择直接运行我们提供的一键脚本，也可以按步骤分别独立运行。

### 方式 1：一键运行完整实验流程
在 `pj1` 目录下，直接执行 `run_experiments.sh`。它会自动按照（爬虫 -> 熵计算 -> 齐夫定律验证）的顺序执行所有逻辑。
```bash
cd pj1
chmod +x run_experiments.sh
./run_experiments.sh
```

### 方式 2：分步骤独立运行

若想分开测试各模块功能，可按以下步骤操作：

#### 步骤 1: 运行爬虫搜集并清洗数据
爬虫会抓取网页内容，并在 `quotes_spider/pipelines.py` 中经过多步清洗：转去 HTML 标签、屏蔽非法乱码，使用中英文不同的正则表达式白名单保留可用文本。
```bash
cd pj1/quotes_spider
chmod +x run_spider.sh
./run_spider.sh
```
*(数据将被划分规模保存在 `../data/` 目录)*

#### 步骤 2: 计算信息熵并导出语料统计
主要基于 `statistics/calc_entropy.py` 进行运算，该脚本：
- **英文处理**：正则提取 `[a-z]+` 单词，计算英文字母和英文单词熵。
- **中文处理**：使用 `jieba` 分词进行词汇切分，计算汉字（字）和中文词汇（词）的概率及熵。
```bash
cd pj1/statistics
chmod +x run_calc.sh
./run_calc.sh
```
该脚本会自动遍历不同规模的文件，将所有结果以及运行统计保存至 `statistics/results/calc_log.txt` 中，方便查阅。

#### 步骤 3: 验证齐夫定律 (Zipf's Law)
使用英文预料的词频数据验证 "词频与排名成反比" 的规律，并绘制相关对数折线图。
```bash
cd pj1/verification
chmod +x run_verify.sh
./run_verify.sh
```
*(生成的图表将保存于 `verification/results/` 中)*

---

## 📝 实验报告

详细的实验记录、语料库多维度统计分析、不同规模预料的对比结果均整理在 `report` 文件夹下的 LaTeX 文档中。
