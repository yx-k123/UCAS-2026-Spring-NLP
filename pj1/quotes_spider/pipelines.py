# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import re
import html

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EN_KEEP_RE = re.compile(r"[^A-Za-z0-9\s.,!?;:'\"()\-]")
# 中文规则中增加了全角破折号 — (以及常用的～等，如果需要也可以再加)
ZH_KEEP_RE = re.compile(r"[^\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9\s，。！？；：、‘’“”（）《》【】\-—]")
MULTI_SPACE_RE = re.compile(r"\s+")


def _normalize_basic(text: str) -> str:
    # 1. 解析 HTML 实体 (如 &nbsp; 转换为空格, &amp; 转换为 &)
    text = html.unescape(text)
    # 2. 去除 HTML 标签与 URL
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    return text


def clean_en_text(text: str) -> str:
    text = _normalize_basic(text)
    # 3. 将不属于白名单的字符替换为空格
    text = EN_KEEP_RE.sub(" ", text)
    # 4. 最后统一合并多个空白字符（包括空格、换行、制表符等）
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def clean_zh_text(text: str) -> str:
    text = _normalize_basic(text)
    # 3. 将不属于白名单的字符替换为空格
    text = ZH_KEEP_RE.sub(" ", text)
    # 4. 最后统一合并多个空白字符
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


class QuotesSpiderPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        title = adapter.get("title", "") or ""
        content = adapter.get("content", "") or ""
        language = (adapter.get("language", "") or "").lower()

        if language == "zh":
            cleaned_title = clean_zh_text(title)
            cleaned_content = clean_zh_text(content)
        else:
            cleaned_title = clean_en_text(title)
            cleaned_content = clean_en_text(content)

        # 过滤清洗后为空或过短的语料，避免污染训练数据。
        if not cleaned_title or not cleaned_content:
            raise DropItem("empty title/content after cleaning")
        if len(cleaned_content) < 30:
            raise DropItem("content too short after cleaning")

        adapter["title"] = cleaned_title
        adapter["content"] = cleaned_content
        adapter["language"] = language or ("zh" if spider.name == "sina" else "en")
        return item
