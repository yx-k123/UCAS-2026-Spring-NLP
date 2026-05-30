# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import time
import re
import urllib3
import json

# 禁用安全警告（因为我们要忽略SSL验证）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 配置区 =================
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Connection': 'keep-alive'
}

TARGET_COUNT = 100

# ================= 辅助函数 =================
def clean_and_split(text):
    if not text: return []
    text = re.sub(r'\s+', ' ', text).strip()
    # 过滤掉一些网页常见杂音
    text = re.sub(r'点击.*?查看|相关阅读|上一篇|下一篇|编辑本段|收藏|查看我的收藏|举报|删除|回复|引用', '', text)
    
    # 标点切分（包含更多口语化标点）
    chunks = re.split(r'([。！？~…])', text)
    sentences = []
    current_sent = ""
    for chunk in chunks:
        current_sent += chunk
        if re.search(r'[。！？~…]', chunk):
            # 过滤掉太短的句子和包含 weird 字符的句子
            if len(current_sent) > 6 and not re.search(r'[<>{}]', current_sent):
                sentences.append(current_sent)
            current_sent = ""
    return sentences

def save_to_utf-8(filename, sentences):
    try:
        # 强制用 utf-8 保存，errors='replace' 把不能编码的字变成问号，防止报错
        with open(filename, 'w', encoding='utf-8', errors='replace') as f:
            for s in sentences:
                f.write(s + '\n')
        print(f"? 文件已保存: {filename} (utf-8, {len(sentences)} 行)")
    except Exception as e:
        print(f"? 保存失败: {e}")

# ================= 1. 爬取笑话网 (幽默口语) =================
def crawl_jokes():
    print("\n--- 正在爬取：笑话网站 (幽默口语) ---")
    sentences = []
    
    base_url = "http://www.haha56.net/xiaohua/youmo/"
    
    try:
        resp = requests.get(base_url, headers=HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        all_links = soup.find_all('a', href=True)
        valid_urls = []
        
        for link in all_links:
            href = link['href']
            if re.search(r'\d+\.html', href) and 'youmo' in href:
                if not href.startswith('http'):
                    if href.startswith('/'):
                        href = "http://www.haha56.net" + href
                    else:
                        href = "http://www.haha56.net/xiaohua/youmo/" + href
                valid_urls.append(href)
        
        valid_urls = list(set(valid_urls))
        print(f"? 扫描到 {len(valid_urls)} 个笑话页面...")

        for url in valid_urls:
            if len(sentences) >= TARGET_COUNT: break
            
            try:
                r = requests.get(url, headers=HEADERS, timeout=5)
                r.encoding = 'utf-8'
                sub_soup = BeautifulSoup(r.text, 'html.parser')
                
                paragraphs = sub_soup.find_all('p')
                full_text = "".join([p.get_text() for p in paragraphs])
                
                new_sents = clean_and_split(full_text)
                for s in new_sents:
                    if s not in sentences:
                        sentences.append(s)
                
                print(f"  > 当前收集: {len(sentences)} 句")
                
            except Exception:
                continue

    except Exception as e:
        print(f"? 笑话网访问失败: {e}")

    save_to_utf-8('data/01_raw/corpus_jokes_utf-8.txt', sentences[:TARGET_COUNT])

# ================= 2. 爬取贴吧 (网络口语) =================
def crawl_tieba():
    print("\n--- 正在爬取：百度贴吧 (网络口语) ---")
    sentences = []
    
    # 贴吧的一些热门话题
    tieba_names = ['搞笑', '闲聊', '段子', '吐槽', '生活', '八卦']
    
    for tieba_name in tieba_names:
        if len(sentences) >= TARGET_COUNT: break
        
        try:
            # 贴吧首页
            url = f"https://tieba.baidu.com/f?kw={tieba_name}"
            resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            resp.encoding = 'utf-8'
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 查找帖子链接
            links = soup.find_all('a', class_=re.compile(r'j_th_tit'))
            
            post_urls = []
            for link in links:
                href = link.get('href', '')
                if href.startswith('/p/'):
                    post_urls.append(f"https://tieba.baidu.com{href}")
            
            print(f"  ? 在 '{tieba_name}' 吧找到 {len(post_urls)} 个帖子...")
            
            for post_url in post_urls[:10]:  # 每个吧爬10个帖子
                if len(sentences) >= TARGET_COUNT: break
                
                try:
                    r = requests.get(post_url, headers=HEADERS, timeout=5, verify=False)
                    r.encoding = 'utf-8'
                    sub_soup = BeautifulSoup(r.text, 'html.parser')
                    
                    # 贴吧内容在 class="d_post_content" 中
                    posts = sub_soup.find_all(['div', 'cc'], class_=re.compile(r'd_post_content|p_content'))
                    
                    for post in posts:
                        text = post.get_text()
                        new_sents = clean_and_split(text)
                        
                        for s in new_sents:
                            if len(s) > 6 and s not in sentences:
                                sentences.append(s)
                    
                    print(f"  > 当前收集: {len(sentences)} 句")
                    
                except Exception:
                    continue
            
            time.sleep(1)
            
        except Exception as e:
            print(f"  ? '{tieba_name}' 吧访问失败: {e}")
            continue
    
    save_to_utf-8('data/01_raw/corpus_tieba_utf-8.txt', sentences[:TARGET_COUNT])

# ================= 3. 爬取知乎 (日常问答) =================
def crawl_zhihu():
    print("\n--- 正在爬取：知乎 (日常问答) ---")
    sentences = []
    
    # 知乎的一些生活话题
    topics = ['日常', '生活', '吐槽', '搞笑', '趣事']
    
    for topic in topics:
        if len(sentences) >= TARGET_COUNT: break
        
        try:
            # 使用知乎搜索
            url = f"https://www.zhihu.com/search?type=content&q={topic}"
            resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            resp.encoding = 'utf-8'
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 提取回答内容
            contents = soup.find_all(['div', 'span'], class_=re.compile(r'RichText|CopyrightRichText'))
            
            for content in contents:
                if len(sentences) >= TARGET_COUNT: break
                
                text = content.get_text()
                new_sents = clean_and_split(text)
                
                for s in new_sents:
                    if len(s) > 8 and len(s) < 80 and s not in sentences:  # 口语化句子不会太长
                        sentences.append(s)
            
            print(f"  > 抓取话题 '{topic}': 当前收集 {len(sentences)} 句")
            time.sleep(1)
            
        except Exception as e:
            print(f"  ? 话题 '{topic}' 失败: {e}")
            continue
    
    save_to_utf-8('data/01_raw/corpus_zhihu_utf-8.txt', sentences[:TARGET_COUNT])

# ================= 4. 爬取微博 (社交媒体口语) =================
def crawl_weibo():
    print("\n--- 正在爬取：微博话题 (社交媒体口语) ---")
    sentences = []
    
    # 微博热搜话题
    hashtags = ['日常', '搞笑', '段子', '吐槽', '生活碎片', '闲聊']
    
    for hashtag in hashtags:
        if len(sentences) >= TARGET_COUNT: break
        
        try:
            # 微博搜索
            url = f"https://s.weibo.com/weibo?q=%23{hashtag}%23"
            resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            resp.encoding = 'utf-8'
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 微博内容通常在特定的 div 中
            posts = soup.find_all(['p', 'div'], class_=re.compile(r'txt|content'))
            
            for post in posts:
                if len(sentences) >= TARGET_COUNT: break
                
                text = post.get_text()
                new_sents = clean_and_split(text)
                
                for s in new_sents:
                    # 微博句子通常比较短小精悍
                    if len(s) > 6 and len(s) < 100 and s not in sentences:
                        sentences.append(s)
            
            print(f"  > 话题 '{hashtag}': 当前收集 {len(sentences)} 句")
            time.sleep(1)
            
        except Exception as e:
            print(f"  ? 话题 '{hashtag}' 失败: {e}")
            continue
    
    save_to_utf-8('data/01_raw/corpus_weibo_utf-8.txt', sentences[:TARGET_COUNT])

# ================= 5. 爬取豆瓣 (影评/书评口语) =================
def crawl_douban():
    print("\n--- 正在爬取：豆瓣短评 (影评书评口语) ---")
    sentences = []
    
    # 豆瓣电影TOP250的短评通常很口语化
    try:
        url = "https://movie.douban.com/top250"
        resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)
        resp.encoding = 'utf-8'
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 获取电影链接
        movie_links = soup.find_all('a', href=re.compile(r'subject/\d+/'))
        
        movie_urls = []
        for link in movie_links:
            href = link.get('href', '')
            if 'subject' in href:
                movie_urls.append(href)
        
        movie_urls = list(set(movie_urls))[:20]  # 限制20部电影
        print(f"  ? 找到 {len(movie_urls)} 部电影...")
        
        for movie_url in movie_urls:
            if len(sentences) >= TARGET_COUNT: break
            
            try:
                # 访问短评页面
                comment_url = movie_url.rstrip('/') + '/comments?status=P'
                r = requests.get(comment_url, headers=HEADERS, timeout=5, verify=False)
                r.encoding = 'utf-8'
                sub_soup = BeautifulSoup(r.text, 'html.parser')
                
                # 短评内容
                comments = sub_soup.find_all('span', class_='short')
                
                for comment in comments:
                    text = comment.get_text()
                    new_sents = clean_and_split(text)
                    
                    for s in new_sents:
                        if len(s) > 6 and s not in sentences:
                            sentences.append(s)
                
                print(f"  > 当前收集: {len(sentences)} 句")
                time.sleep(1)
                
            except Exception:
                continue
        
    except Exception as e:
        print(f"? 豆瓣访问失败: {e}")
    
    save_to_utf-8('data/01_raw/corpus_douban_utf-8.txt', sentences[:TARGET_COUNT])

# ================= 6. 爬取小红书风格 (生活分享口语) =================
def crawl_lifestyle():
    print("\n--- 正在爬取：天涯论坛 (生活分享口语) ---")
    sentences = []
    
    # 天涯的情感、生活板块
    base_url = "http://bbs.tianya.cn"
    
    try:
        boards = [
            "http://bbs.tianya.cn/list-feeling-1.shtml",  # 情感天地
            "http://bbs.tianya.cn/list-funinfo-1.shtml",   # 娱乐八卦
            "http://bbs.tianya.cn/list-free-1.shtml"       # 天涯杂谈
        ]
        
        post_urls = []
        
        for board_url in boards:
            if len(post_urls) >= 40: break
            
            try:
                resp = requests.get(board_url, headers=HEADERS, timeout=10, verify=False)
                resp.encoding = 'utf-8'
                
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                links = soup.find_all('a', href=re.compile(r'post-'))
                
                for link in links:
                    href = link['href']
                    if not href.startswith('http'):
                        href = base_url + href
                    post_urls.append(href)
                
                time.sleep(1)
                
            except Exception:
                continue
        
        post_urls = list(set(post_urls))[:40]
        print(f"  ? 发现 {len(post_urls)} 个帖子...")
        
        for url in post_urls:
            if len(sentences) >= TARGET_COUNT: break
            
            try:
                r = requests.get(url, headers=HEADERS, timeout=5, verify=False)
                r.encoding = 'utf-8'
                sub_soup = BeautifulSoup(r.text, 'html.parser')
                
                posts = sub_soup.find_all(['div', 'p'], class_=re.compile(r'bbs-content|atl-item'))
                
                for post in posts:
                    text = post.get_text()
                    new_sents = clean_and_split(text)
                    
                    for s in new_sents:
                        if len(s) > 6 and s not in sentences:
                            sentences.append(s)
                
                print(f"  > 当前收集: {len(sentences)} 句")
                
            except Exception:
                continue
        
    except Exception as e:
        print(f"? 天涯论坛访问失败: {e}")
    
    save_to_utf-8('data/01_raw/corpus_lifestyle_utf-8.txt', sentences[:TARGET_COUNT])

if __name__ == "__main__":
    print("="*50)
    print("开始爬取口语化汉语文本语料")
    print("="*50)
    
    # 1. 笑话（幽默口语）
    crawl_jokes()
    print("-" * 50)
    time.sleep(2)
    
    # 2. 贴吧（网络口语）
    crawl_tieba()
    print("-" * 50)
    time.sleep(2)
    
    # 3. 知乎（日常问答）
    crawl_zhihu()
    print("-" * 50)
    time.sleep(2)
    
    # 4. 微博（社交媒体）
    crawl_weibo()
    print("-" * 50)
    time.sleep(2)
    
    # 5. 豆瓣（影评书评）
    crawl_douban()
    print("-" * 50)
    time.sleep(2)
    
    # 6. 天涯（生活分享）
    crawl_lifestyle()
    
    print("\n" + "="*50)
    print("? 所有口语化语料爬取完成！")
    print("="*50)