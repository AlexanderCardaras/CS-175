# scrapy runspider raw_data.py -o ../../res/raw_data.json

import scrapy


class QuotesSpider(scrapy.Spider):
    name = 'raw_data'

    custom_settings = {
        "DOWNLOAD_DELAY": "1.0",  # 1000 ms of delay
    }

    def start_requests(self):
        urls = [
            "https://www.law.cornell.edu/uscode/text",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):

        # Find title links
        content = response.css('div.tab-pane.active')
        for line in content.css('toc').css('ol.list-unstyled').css('li.tocitem'):
            next_page = line.css('a::attr("href")').get()
            if next_page is not None:
                yield response.follow(next_page, self.parse)

        # Find chapter links
        content = response.css('div.toc.title')
        for line in content.css('ol.list-unstyled').css('li.tocitem'):
            next_page = line.css('a::attr("href")').get()
            if next_page is not None:
                yield response.follow(next_page, self.parse)

        # Find text links
        content = response.css('div.toc.chapter')
        for line in content.css('ol.list-unstyled').css('li.tocitem'):
            next_page = line.css('a::attr("href")').get()
            if next_page is not None:
                yield response.follow(next_page, self.parse)

        # Get text
        content = response.css('div.content')
        for line in content.css('p'):
            yield {
                'text': line.get()
            }

        # Go to next link
        next_page = response.css('li.next a::attr("href")').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
