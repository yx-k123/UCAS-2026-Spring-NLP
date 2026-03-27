import scrapy

class BBCSpider(scrapy.Spider):
    name = "bbc"
    start_urls = [
        'https://www.bbc.com/news',
        'https://www.bbc.com/news/world',
        'https://www.bbc.com/news/business',
        'https://www.bbc.com/news/technology',
        'https://www.bbc.com/news/science_and_environment',
        'https://www.bbc.com/news/entertainment_and_arts',
    ]

    def parse(self, response):
        # 优化选择器以匹配更多链接
        links = response.css('a::attr(href)').getall()
        
        for link in links:
            # 补全相对路径
            if link.startswith('/'):
                link = 'https://www.bbc.com' + link
            
            # 扩大抓取范围，只要是新闻文章页就跟进
            # BBC 文章通常包含 8 位数字 ID，例如 /news/world-68666666
            if '/news/' in link and any(char.isdigit() for char in link):
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
