from collections import deque
import time
import re

import requests
from bs4 import BeautifulSoup
import os
from os import path

class BaseCrawler:
    """
        Ideally we could use this base crawler class to support any of the crawlers we write for this project.
    """
    defaults = dict(
        delay=2,
        web_cache_root = "web/"
    )

    def __init__(self, **kwargs):
        #Only add attributes that we define in the defaults class dictionary
        for key in BaseCrawler.defaults:
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, BaseCrawler.defaults[key])
            
        self.discovered = set()
        #queue for visiting pages
        self.visit_queue = deque()
    
    #Note: I assume we only want to scrape a single domain with one crawler.
    def crawl(self, domain, start_page):
        """ The heavy duty method for the crawler. The crawler will crawl a website until it runs out of URLs to process.
        """
        #Ensure we have a folder to put cache the crawled pages in
        if not os.path.exists(self.web_cache_root):
            os.makedirs(self.web_cache_root)
    
        self.discovered.add(start_page)
        self.visit_queue.append(start_page)
        
        while self.visit_queue:
            current_path = self.visit_queue.popleft()
            
            #Construct paths to the cache file and web url
            filepath = self.web_cache_root + current_path
            if filepath[-1] == '/':
                filepath = filepath[:-1]
            #If the path is missing an extension it will collide with folder names
            # so we add a .html extension
            if filepath.rfind('.') <= filepath.rfind("/"):
                # we need == in the case that there are no . or / so rfind returns -1 for both
                filepath += ".html"
            url = domain + current_path
            
            html = self.load_resource(filepath, url)
            if html:
                self.discover_pages(html)
    
    def load_resource(self, filepath, url):
        """
            Local an html document we want to crawl either from a local file-system class (ideal) or if the cache doesn'take
            exist, from the web. This will cache any resources loaded from the webt so that the next time they will be 
            available locally.
        """
        if not path.exists(filepath):
            print("Sleeping for {} seconds".format(self.delay))
            time.sleep(self.delay)
            response = requests.get(url)
            
            if not response.ok:
                print(f"Error: GET {url} returned {response.status_code} {response.reason}")
                return None
            else:
                print(f"GET {url} -> {response.status_code} {response.reason}")
                
            #Make necessary folders for the file if the required folders in the path don't exist
            last_slash = filepath.rfind('/')
            if last_slash != -1:
                folders = filepath[:last_slash+1]
            if not os.path.exists(folders):
                os.makedirs(folders)
                
            with open(filepath, "x", encoding="utf-8") as f:
                f.write(response.text)
            f.close()
            return response.text
        else:
            print(f"Reading file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    
    def discover(self, url):
        """
            Add a URL to our queue for visitng URLs and add to the set of visited urls
        """
        if not url in self.discovered:
            self.discovered.add(url)
            self.visit_queue.append(url)
    
    def discover_pages(self, html):
        """
            Abstract method left up to subclasses to implement. They should take the html text and parse what links are
            relevant to the crawler. After they have the links they should call self.discover(url).
        """
        raise NotImplementedError("Subclasses of BaseCrawler must implement discover_pages(html) !")
            

class ScripturesBYUCrawler(BaseCrawler):
    """
        The way scriptures.byu.edu works is its a single webpage that loads different content using Asynchronous JavaScript
        Requests (AJAX). All the links on the webpage are thus not to new URLs but to javascript functions that internally
        load new content. However, through snooping around I found their Javascript got resources that were essentially
        pieces of HTML code on a public HTTP GET API, so all I have to do is find a way to change the JavaScript function
        calls into URLs to load the same textual content the JavaScript would, or use Selenium to simulate clicking the
        links and thus triggering new content loads. I have opted to go with the former, believing it to be quicker.
        
        As this webpage is so unique, I wrote a very specialiazed crawler for it.
    """
    def __init__(self, **kwargs):
        if "web_cache_root" not in kwargs:
            kwargs["web_cache_root"] = "web/scriptures.byu.edu/"
        super().__init__(**kwargs)
        
    def parse_js_link(self, js_onclick):
        if not js_onclick:
            return None
        elif js_onclick.startswith('getConf('):
            #The link looks like: "getConf('2020', 'O')" so splitting by the ' character would give an array like
            # pieces[0] = "getConf(", pieces[1] = "2020", pieces[2] = ", ", pieces[3] = "O", pieces[4] = ")"
            pieces = js_onclick.split('\'')
            
            if (len(pieces) == 5 and pieces[1].isdigit() and 
                (pieces[3] == "O" or pieces[3] == "A")): #sanity check
                #Example URL: citation_index/gc_ajax/2020/O
                return "citation_index/gc_ajax/" + pieces[1] + "/" + pieces[3]
            else:
                print(f"Possible bug: couldn't parse '{js_onclick}'")
                
        elif js_onclick.startswith('getTalk('):
            #The link looks like: "getTalk('8460');" so splitting by the ' character would give an array like
            #pieces[0] = "getTalk(", pieces[1] = "8460", pieces[2] = ");"
            pieces = js_onclick.split('\'')
            
            if len(pieces) == 3 and pieces[1].isdigit(): #sanity check
                #Example URL: content/talks_ajax/8460/
                return "content/talks_ajax/"+pieces[1]+"/"
            else:
                print(f"Possible bug: couldn't parse '{js_onclick}'")
        
    def discover_pages(self, html):
        soup = BeautifulSoup(html, "html.parser")
        
        #All the links we are interested in are inside this div
        content_div = soup.find(attrs={"class": "nano-content"})
        if content_div:
            for a_tag in content_div.find_all('a'):
                if "onclick" in a_tag.attrs:
                    link = self.parse_js_link(a_tag["onclick"])
                    if link:
                        self.discover(link)        
        
if __name__ == "__main__":
    ScripturesBYUCrawler().crawl("https://scriptures.byu.edu/", "citation_index/gc_ajax")
    