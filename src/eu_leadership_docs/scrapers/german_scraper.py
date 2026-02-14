import requests
import csv
import time
from bs4 import BeautifulSoup
from eu_leadership_docs.utils.path_helpers import raw_path
from pathlib import Path

BASE_URL = "https://www.auswaertiges-amt.de"
OUTPUT_CSV = raw_path("aa_press_2014_2025_full_with_2025.csv")

# każdy rok
years_URLs = {
    '2014':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2014-archivpressemitteilungen-node/216810-216810?limit=25', 
    '2015':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2015-archivpressemitteilungen-node/216806-216806?limit=25',
    '2016':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2016-archivpressemitteilungen-node/216802-216802?limit=25', 
    '2017':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2017-archivpressemitteilungen-node/1611734-1611734?limit=25',
    '2018':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2018-archivpressemitteilungen-node/2471814-2471814?limit=25', 
    '2019':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2019-archivpressemitteilungen-node/2471820-2471820?limit=25',
    '2020':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2020-archivpressemitteilungen-node/2471824-2471824?limit=25', 
    '2021':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2021-archivpressemitteilungen-node/2509530-2509530?limit=25',
    '2022':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2022-archivpressemitteilungen-node/2577098-2577098?limit=25', 
    '2023':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2023-archivpressemitteilungen-node/2642166-2642166?limit=25',
    '2024':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2024-archivpressemitteilungen-node/2698876-2698876?limit=25',
    '2025':'https://www.auswaertiges-amt.de/ajax/json-filterlist/de/newsroom/presse/web-archiv/archiv-pressemitteilungen/2025-archivpressemitteilungen-node/2755112-2755112?limit=25'
}

def get_json_results(url, offset=0):
    if "offset=" not in url:
        full_url = f"{url}&offset={offset}"
    else:
        full_url = url.split("&offset=")[0] + f"&offset={offset}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(full_url, headers=headers)
    if r.status_code != 200:
        print(f"ERROR:( {full_url}")
        return None
    return r.json()


def parse_article(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"ATTENTION, I cannot scrape this URL: {url}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    for selector in [
        ("div", {"class": "c-rte--default", "data-css": "c-rte"}),
        ("div", {"class": "richtext"}),
        ("div", {"class": "text"})
    ]:
        div = soup.find(*selector)
        if div:
            return div.get_text(" ", strip=True)
    return ""


def scrape_year(year, url, writer):
    print(f"\n!!! Downloading year {year} !!!")
    total = 0
    offset = 0
    limit = 25

    while True:
        data = get_json_results(url, offset=offset)
        if not data or not data.get("items"):
            print("No more items")
            break

        items = data["items"]
        for item in items:
            link = BASE_URL + item.get("link", "").strip()
            title = (item.get("headline") or "").strip()
            doc_type = (item.get("name") or "").strip()
            date = (item.get("date") or "").strip()
            snippet = (item.get("text") or "").strip()
            full_text = parse_article(link)

            writer.writerow({
                "year": year,
                "date": date,
                "type": doc_type,
                "title": title,
                "snippet": snippet,
                "url": link,
                "full_text": full_text
            })

            print(f"!!! [{year}] {title[:80]}")
            total += 1
            time.sleep(0.2)

        offset += limit

    print(f"!!! Year {year} finished, we scraped {total} records !!!\n")


# run
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["year", "date", "type", "title", "snippet", "url", "full_text"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for year, url in years_URLs.items():
        scrape_year(year, url, writer)

print(f"\nready! Saved in the following file: {OUTPUT_CSV}")
