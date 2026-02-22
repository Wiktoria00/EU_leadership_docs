from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin
from eu_leadership_docs.utils.helpers import raw_path


# ARCHIVE URL (for older press releases, until 2017)
START_URL = (
    "https://www.eeas.europa.eu/filter-page/archive_en?"
    "fulltext=Russia&created_from=2014-01-01&created_to=2017-07-07&"
    "f%5B0%5D=arch_ct%3Aeeas_press&page={}")

BASE_URL = "https://www.eeas.europa.eu"
OUTPUT_CSV = raw_path("eeas_until_2017.csv")



def safe_goto(page, url, max_retries=5, base_delay=5):
    retries = 0
    while retries <= max_retries:
        try:
            response = page.goto(url, timeout=60000)
            if response.status == 429:
                wait_time = base_delay * (2 ** retries)  # rosnące opóźnienie
                print(f"429 Too Many Requests. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                continue
            return response
        except Exception as e:
            print(f"Error during page.goto: {e}")
            wait_time = base_delay * (2 ** retries)
            print(f"Retrying after {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception(f"Failed to load page {url} after {max_retries} retries.")


def extract_date_by_clock(soup, icon_class):
    icon = soup.select_one(f"i.{icon_class}")
    if not icon:
        return None
    node = icon.next_sibling
    while node and (not hasattr(node, "strip") or not node.strip()):
        node = node.next_sibling
    if node:
        return node.strip().split("—")[0].strip()
    return None

def scrape_page(page, page_num):
    url = START_URL.format(page_num)
    print(f"Scraping: {url}")
    safe_goto(page, url, max_retries=5, base_delay=5)
    time.sleep(5)
    soup = BeautifulSoup(page.content(), "html.parser")
    articles = soup.select("div.card")
    
    results = []
    for art in articles:
        link_tag = art.select_one("h3[class*='card-title'] a[href]")  # updated
        if not link_tag:
            continue
        title = link_tag.get_text(strip=True)
        article_url = link_tag["href"]
        if article_url.startswith("/"):
            article_url = urljoin(BASE_URL, article_url)

        date = extract_date_by_clock(art, "e-clock")
        results.append((title, date, article_url))
    return results


def parse_article(page, url):
    if url.lower().endswith(".pdf") or "/doc/document/" in url:
        print(f"Direct PDF link: {url}")
        return "", url, None

    print(f"Visiting {url}")
    safe_goto(page, url, max_retries=5, base_delay=5)
    time.sleep(5)
    soup = BeautifulSoup(page.content(), "html.parser")

    paragraphs = []
    if "eeas.europa.eu" in url:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("article p, div.field--item p")]
    elif "consilium.europa.eu" in url:
        consilium_selectors = [
            "div.c-article__body p", "div.c-article__body",
            "div.c-article__text p", "div.c-article__text",
            "div.textblock p", "div.rte p", "main#content p",
            "div.gsc-bge-grid__area"
        ]
        for sel in consilium_selectors:
            elems = soup.select(sel)
            for e in elems:
                ps = e.find_all("p")
                if ps:
                    paragraphs.extend([p.get_text(" ", strip=True) for p in ps])
                else:
                    paragraphs.append(e.get_text(" ", strip=True))
    elif "ec.europa.eu" in url:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("div.ecl-content-block__description")]
    else:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]

    text = " ".join(paragraphs) if paragraphs else ""

    pdf_tag = soup.select_one("a.gsc-link.pa-download-document-click, a[href$='.pdf']")
    pdf_url = None
    if pdf_tag:
        pdf_url = pdf_tag.get("href")
        if pdf_url and pdf_url.startswith("/"):
            pdf_url = urljoin(url, pdf_url)

    article_date = extract_date_by_clock(soup, "e-wall-clock3")
    return text, pdf_url, article_date

all_data = []
visited_urls = set()
page_num = 0

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()

    while True:
        articles = scrape_page(page, page_num)
        if not articles:
            print("No more results on this page.")
            break

        new_count = 0
        for title, date, url in articles:
            if url in visited_urls:
                continue
            visited_urls.add(url)
            new_count += 1

            try:
                text, pdf, article_date = parse_article(page, url)
            except Exception as e:
                print(f"Error parsing {url}: {e}")
                text, pdf, article_date = None, None, None

            all_data.append({
                "title": title,
                "listing_date": date,
                "article_date": article_date,
                "article_url": url,
                "pdf_url": pdf,
                "text": text,
            })

        if new_count == 0:
            print("No new articles found, stopping.")
            break

        page_num += 1
        time.sleep(2)

    browser.close()

df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} records to {OUTPUT_CSV}")