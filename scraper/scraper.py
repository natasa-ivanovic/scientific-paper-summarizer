from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
import requests
import math
import time
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


# configure webdriver
options = Options()
options.headless = True  # hide GUI
options.add_argument("--window-size=1920,1080")  # set window size to native GUI size
options.add_argument("start-maximized")  # ensure window is full-screen

# configure chrome browser to not load images and javascript
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option(
    # this will disable image loading
    "prefs", {"profile.managed_default_content_settings.images": 2}
)

start_time = time.time()

service = Service(executable_path="chromedriver.exe")

driver = webdriver.Chrome(options=options, chrome_options=chrome_options, service=service)

ftn_links = ['http://www.ftn.uns.ac.rs/ojs/index.php/zbornik/issue/archive?issuesPage=1#issues']
             # 'http://www.ftn.uns.ac.rs/ojs/index.php/zbornik/issue/archive?issuesPage=2#issues']


zbornik_list_of_links = []
print('~~ Starting scrape of zbornik links ~~')

for link in ftn_links:
    driver.get(link)
    element = WebDriverWait(driver=driver, timeout=5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class=obj_issue_summary]'))
    )
    # print(driver.page_source)
    all_sections = driver.find_elements(By.CSS_SELECTOR, 'div.obj_issue_summary a.title')
    for section in all_sections:
        zb_link = section.get_attribute('href')
        zbornik_list_of_links.append(zb_link)

print('~~ Scrape of zbornik links successful ~~')

list_of_radovi = []

print('~~ Starting scrape of rad links ~~')
for i, zbornik_link in enumerate(zbornik_list_of_links):
    print("\r[{0:50s}] {1:.1f}% - ({3} of {4}) - scraping from {2} ".format(
        "#" * int(math.ceil((i + 1) / len(zbornik_list_of_links) * 50)),
        100 * (i + 1) / len(zbornik_list_of_links),
        zbornik_link,
        i + 1,
        len(zbornik_list_of_links)
    ), end="")
    driver.get(zbornik_link)
    element = WebDriverWait(driver=driver, timeout=10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'a[class="obj_galley_link pdf"]'))
    )
    # print(driver.page_source)
    all_radovi = driver.find_elements(By.CSS_SELECTOR, 'div[class="obj_article_summary"]')

    for rad_index, rad_el in enumerate(all_radovi[1:]):
        rad_ime = 'Rad - ' + str(rad_index)
        try:
            rad_ime_el = rad_el.find_element(By.CSS_SELECTOR, 'div.title a')
            rad_ime = slugify(rad_ime_el.text, True)
            # add / na kraju linka i replace view sa download
            rad_link_el = rad_el.find_element(By.CSS_SELECTOR, 'ul[class="galleys_links"] li a')
            rad_link = rad_link_el.get_attribute('href') + '/'
            rad_link = rad_link.replace('view', 'download')
            list_of_radovi.append((rad_ime, rad_link))
        except NoSuchElementException:
            print("\n~!~ Couldn't find element with name " + rad_ime)

print('~~ Scrape of rad links successful ~~')
# chatGPT magic for downloading
print('~~ Starting download ~~')
for i, rad in enumerate(list_of_radovi):
    rad_name, rad_link = rad

    response = requests.get(rad_link)
    # print('Downloading file with name ', rad_name)
    with open('radovi/' + rad_name + '.pdf', "wb") as f:
        f.write(response.content)
    # chatgpt magic progress bar after download is done
    print("\r[{0:50s}] {1:.1f}% - ({3} of {4}) - {2} ".format(
        "#" * int(math.ceil((i + 1) / len(list_of_radovi) * 50)),
        100 * (i + 1) / len(list_of_radovi),
        rad_name,
        i + 1,
        len(list_of_radovi)
    ), end="")
print('~~ Download successful ~~')

end_time = time.time()
elapsed_time = end_time - start_time
print("Process complete in - {:.2f} seconds".format(elapsed_time))

driver.quit()
