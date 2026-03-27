import scrapy
import json

class SinaSpider(scrapy.Spider):
    name = "sina"
    # 新浪新闻常使用动态加载，这里尝试从其新闻首页或特定的 API 开始
    start_urls = ['https://news.sina.com.cn/']

    def parse(self, response):
        # 提取导航栏或列表页中的新闻链接
        links = response.css('a::attr(href)').getall()
        for link in links:
            if 'news.sina.com.cn' in link and link.endswith('.shtml'):
                yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        # 简单提取文章标题和正文
        title = response.css('h1.main-title::text').get() or response.css('h1#main_title::text').get()
        content = "".join(response.css('div.article p::text').getall()) or "".join(response.css('div#article p::text').getall())
        
        if title and content:
            yield {
                'url': response.url,
                'title': title.strip(),
                'content': content.strip(),
                'language': 'zh'
            }
