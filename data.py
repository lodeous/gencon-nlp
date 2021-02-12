from urllib import robotparser
from selenium import webdriver

rp = robotparser.RobotFileParser("https://churchofjesuschrist.org/robots.txt")
rp.read()


def crawl_topics(main_page="https://www.churchofjesuschrist.org/general-conference/topics?lang=eng"):
    assert rp.can_fetch("*", main_page)

    topics, links = [], []

    browser = webdriver.Chrome()
    browser.get("https://www.churchofjesuschrist.org/general-conference/topics?lang=eng")
    tile_titles = browser.find_elements_by_class_name("lumen-tile__title")
    for tile in tile_titles:
        a = tile.find_element_by_tag_name("a")
        topics.append(a.text)
        links.append(a.get_attribute("href"))

    browser.close()

    return topics, links


if __name__ == "__main__":
    result = crawl_topics()
    print(list(zip(*result)))
