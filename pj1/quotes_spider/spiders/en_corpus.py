import scrapy

class BBCSpider(scrapy.Spider):
    name = "bbc"
    start_urls = [
        'https://www.bbc.com/news',
    ]

    def parse(self, response):
        # 提取 BBC 新闻主页的文章链接
        links = response.css('a.qa-story-image-link::attr(href)').getall() or \
                response.css('a[data-testid="internal-link"]::attr(href)').getall()
        
        for link in links:
            # 补全相对路径
            if link.startswith('/'):
                link = 'https://www.bbc.com' + link
            
            # 只抓取新闻文章页（排除视频等）
            if '/news/' in link and not link.endswith('/news'):
                yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        # 提取标题
        title = response.css('h1#main-heading::text').get() or \
                response.css('h1::text').get()
        
        # 提取正文主体部分的段落
        content_nodes = response.css('div[data-component="text-block"] p::text').getall() or \
                        response.css('article p::text').getall()
        content = " ".join([p.strip() for p in content_nodes if p.strip()])
        
        if title and content:
            yield {
                'url': response.url,
                'title': title.strip(),
                'content': content,
                'language': 'en'
            }
