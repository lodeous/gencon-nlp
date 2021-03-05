from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

class ConferenceTalkScraper:
    """
    This scraper goes through all the general conference talk files crawled and downloaded by the ScripturesBYUCrawler
    (see crawlers.py) and extracts the main text and metadata and puts it into an easy-to-read format. The metadata is
    saved alongside the main text file (but in a separate .txt file) and is also stored in an aggregated CSV format
    that is readily loaded into a Pandas DataFrame
    """
    sup_tag_matcher = re.compile(r"\<sup.*\<\/sup\>")

    def __init__(self, base_path, out_base_path):
        """
        Creates a new ConferenceTalkScraper that will scrape the HTML files located in base_path directory and output 
        easy-to-read txt files and metadata to the out_base_path directory
        
        Inputs:
            base_path (str) - the path to look for HTML files in
            out_base_path (str) - the path to save the conference text and metadata
        """
        self.base_path = base_path
        self.out_base_path = out_base_path
        self.summary_dict = {}
        
    def scrape(self):
        """
        Automatically scans the base_path directory for HTML files and then scrapes them, outputting the results.
        """
        #Ensure we have a place for the scraped text to go
        if not os.path.exists(self.out_base_path):
            os.makedirs(self.out_base_path)
    
        with os.scandir(self.base_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".html"):
                    html = None
                    with open(entry.path, encoding='utf-8') as f:
                        html = f.read()
                    if html:
                        self._process_html(entry.name, html)
                
        self._save_summary_dict()
    
    #For testing: I made this so I could test scraping on 2-3 files instead of all of them
    def _process_file(self, filename):
        path = self.base_path + filename
        with open(path, encoding='utf-8') as f:
            html = f.read()
        if html:
            self._process_html(str(path).split('/')[-1], html)
            
    def _process_html(self, filename, html):
        """
        Takes the preloaded html and extracts the following information
         - Year
         - Month
         - Speaker
         - Title
         - Kicker (the tagline that appears before the talk, only available on some talks)
         and of course, the main text of the conference talk.
        This method ignores footnotes and references, which are not useful to our analysis.
        After extracting this data it calls _export_data() with the extracted data to save the results
            
        Inputs:
            filename (str) - the filename the HTML was loaded from (used to extract the content_id)
            html (str) - the full HTML text loaded from the file
        """
        content_id = filename[:-5]
        
        #Skip processing files that have already been processed
        #export_path = self.out_base_path + content_id + ".txt"
        #if os.path.exists(export_path):
        #    return
    
        print("Processing:", filename)
    
        soup = BeautifulSoup(html, "html.parser")
        talklabel = soup.find(id='talklabel')
        
        #talklabel.text looks something like
        #'1942–A:2, Heber J. Grant, Personal Testimony of the Lord’s Providence'
        #pieces[0] = '1942–A:2', pieces[1] = 'Heber J. Grant', pieces[2] = 'Personal Testimony of the Lord’s Providence'
        #We can only split a max of 2 times on commas otherwise we will split the title if it has any commas in it.
        pieces = talklabel.text.split(', ', 2)
        
        try:
            year = int(pieces[0][:4])
            month = pieces[0][5]
            speaker = pieces[1]
            
            #Unfortunately sometimes the name has a comma in it, specifically "J Reuben Clark, Jr." (and there is one name
            # with ", Sr." in it too. In order to detect this we must check if the third character is a '.' and then move 
            # the substring from the title to the name
            if pieces[2][2] == '.': #Jr. or Sr.
                speaker += ", " + pieces[2][:3]
                title = pieces[2][5:] #get rid the first 5 characters: "Jr., " or "Sr., "
            else:
                title = pieces[2]
        except:
            print(f"Error in {filename}: couldn't parse textlabel with content: '{textlabel.text}'")
            return
        
        #older talks
        gcbody = soup.find(attrs={"class": "gcbody"})
        #newer talks
        primary = soup.find(id="primary")
        
        if gcbody:
            self._export_data(content_id, year, month, speaker, title, gcbody.text)
        elif primary:
            #The kicker is a line they take from the talk and emphasize at the beginning
            if primary.blockquote and primary.blockquote.div:
                kicker = primary.blockquote.div.text
            else:
                kicker = None
            
            #We need to get rid of the <sup> tags as they contain references. We only care about the text.
            raw_html = str(primary)
            filtered = re.sub(ConferenceTalkScraper.sup_tag_matcher, "", raw_html)
            
            #We want to filter out all the text except the main body of the talk (i.e. ignoring footnotes and the kicker
            #The main body happens to have <p> tags that all have a 'uri' attribute
            text = ""
            filtered_soup = BeautifulSoup(filtered, "html.parser")
            for p_child in filtered_soup("p"):
                if 'uri' in p_child.attrs:
                    text += p_child.text + "\n"
            self._export_data(content_id, year, month, speaker, title, text, kicker)
            
        else:
            print(f"Error in {filename}: couldn't find main text!")
        
    def _export_data(self, content_id, year, month, speaker, title, text, kicker=None):
        """
        Saves the main text of the talk in files named [content_id].txt, and the talk metadata in a seperate file
        named [content_id].meta.txt
        Additionally the metadata is aggregated into a dictionary so it can be saved all together at the end.
            
        Inputs:
            content_id (str) - the content_id of the talk (derived from its unique webpage content_id)
            year (int) - the year the talk was given
            month (str) - either "A" or "O" signifying the talk was given in either April or October
            speaker (str) - the name of the speaker who gave the talk
            text (str) - the full text of the talk
            kicker (str|None) - the tagline presented at the front of the talk which carries some of its main points*
            
            *Not all talks have a kicker
        """
        export_meta_path = self.out_base_path + content_id + ".meta.txt"
        export_path = self.out_base_path + content_id + ".txt"
        
        self.summary_dict[content_id] = [year, month, speaker, title, export_path, kicker]
        
        with open(export_meta_path, 'w', encoding='utf-8') as f:
            f.write(f"YEAR: {year}\n")
            f.write(f"MONTH: {month}\n")
            f.write(f"SPEAKER: {speaker}\n")
            f.write(f"TITLE: {title}\n")
            f.write(f"KICKER: {kicker}\n")
            
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
    
    def _save_summary_dict(self):
        column_names = ['Year', 'Month', 'Speaker', 'Title', 'File', 'Kicker']
        df = pd.DataFrame.from_dict(self.summary_dict, orient='index', columns=column_names)
        df.to_csv(self.out_base_path+"summary.csv")

def test_scraper():
    scraper = ConferenceTalkScraper("web/scriptures.byu.edu/content/talks_ajax/", "data/")
    #Test processing one old type and one new type file.
    scraper._process_file("1.html")
    scraper._process_file("8354.html")
    #Test talk with J Reuben Clark, Jr. as speaker
    scraper._process_file("100.html")
   
if __name__ == "__main__":
    #test_scraper()
    ConferenceTalkScraper("web/scriptures.byu.edu/content/talks_ajax/", "data/").scrape()