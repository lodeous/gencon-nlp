from urllib import robotparser
from selenium import webdriver
import requests
import os
from bs4 import BeautifulSoup

rp = robotparser.RobotFileParser("https://churchofjesuschrist.org/robots.txt")
rp.read()


def crawl_topics(
    main_page="https://www.churchofjesuschrist.org/general-conference/topics?lang=eng",
):
    """
    Get a list of links for pages of General Conference talks by topic
    """

    assert rp.can_fetch("*", main_page)

    links = []

    browser = webdriver.Chrome()
    browser.get(main_page)
    tile_titles = browser.find_elements_by_class_name("lumen-tile__title")
    for tile in tile_titles:
        a = tile.find_element_by_tag_name("a")
        links.append(a.get_attribute("href"))

    browser.close()

    return links


def retrieve_pages(links, parent_dir="./web/churchofjesuschrist.org"):
    """
    Download General Conference talks, with a directory for each topic
    """

    for link in links:
        assert rp.can_fetch("*", link)

        bs = BeautifulSoup(requests.get(link).text, "html.parser")
        topic = bs.find("h1", class_="title").text.strip()

        # create a new directory for this topic
        base_dir = parent_dir + "/" + topic
        os.mkdir(base_dir)

        tile_titles = bs.find_all("div", class_="lumen-tile__title")
        for i, tile in enumerate(tile_titles):
            talk_url = "https://www.churchofjesuschrist.org" + tile.find("a")["href"]
            content = requests.get(talk_url).content
            with open(f"{base_dir}/{i}.html", "wb") as f:
                f.write(content)


if __name__ == "__main__":
    retrieve_pages(crawl_topics())
