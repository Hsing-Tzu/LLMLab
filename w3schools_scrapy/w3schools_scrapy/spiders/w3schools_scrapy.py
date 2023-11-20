import scrapy
import re

class W3SchoolsSpider(scrapy.Spider):
    # 定義一個名為 w3schools 的 Scrapy 爬蟲
    name = "w3schools"
    # 設定開始的 URL
    start_urls = ["https://www.w3schools.com/python/default.asp"]
    # 設定爬蟲
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'FEEDS': {
            'w3schools_scrapy.csv': {
                'format': 'csv',
                'fields': ['url', 'tutorial_name', 'text'],
            },
        },
    }

    def parse(self, response):
        # 迴圈遍歷範圍內的元素
        for i in range(2, 7):
            # 建立 XPath
            xpath = f'//*[@id="leftmenuinnerinner"]/a[{i}]/@href'
            print(xpath)
            
            # 提取連結並追蹤
            tutorial_link = response.urljoin(response.xpath(xpath).get())
            print(tutorial_link)
            yield scrapy.Request(tutorial_link, callback=self.parse_tutorial)

    def parse_tutorial(self, response):
        # 從 URL 中提取相關部分
        pattern = r"\/([a-z_]+)\.asp"
        match = re.search(pattern, response.url)
        tutorial_name = match.group(1) if match else "No match found"

        # 爬取內容
        content = response.xpath('//div[@class="w3-col l10 m12"]')
        text_paragraphs = content.xpath('.//p/text()').getall()
        examples = content.xpath('.//div[@class="w3-example"]/text()').getall()

        # 清理並格式化爬取的文本
        formatted_text = "\n".join(text_paragraphs).strip()
        formatted_examples = "\n".join([re.sub(r'\s+', ' ', ex).strip() for ex in examples])

        # 回傳爬取的數據
        yield {
            'url': response.url,
            'tutorial_name': tutorial_name,
            'text': formatted_text,
            'examples': formatted_examples,
        }

    def close(self, reason):
        pass
