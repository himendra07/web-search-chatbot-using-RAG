import asyncio
import aiohttp
import os
import urllib.parse
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from config import REQUESTS_HEADER
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_url_to_filename(url: str) -> str:
    return urllib.parse.quote(url, safe="")

def decode_filename_to_url(filename: str) -> str:
    return urllib.parse.unquote(filename)

def get_urls(query: str, num_results: int, provider: str = "duckduckgo") -> list[str]:
    provider = (provider or "duckduckgo").lower()
    urls = []

    if provider == "duckduckgo":
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=num_results, region="us-en")
            urls = [item.get("href") for item in results if item.get("href")]
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            urls = []
    elif provider == "google":
        try:
            from googlesearch import search as google_search
            for u in google_search(query, num_results=num_results, lang="en", region="us"):
                if u:
                    urls.append(u)
        except Exception as e:
            logger.warning("Google search failed or was blocked: %s", e)
            urls = []
    else:
        logger.warning("Unknown provider '%s' — returning empty list", provider)
        urls = []
        
    seen = set()
    clean = []
    for u in urls:
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            continue
        if u in seen:
            continue
        seen.add(u)
        clean.append(u)
    return clean

async def fetch_and_save(session: aiohttp.ClientSession, url: str, folder: str) -> str | None:
    filename = encode_url_to_filename(url)
    filepath = os.path.join(folder, filename)

    try:
        async with session.get(url, timeout=20) as response:
            response.raise_for_status()
            content = await response.text()
            soup = BeautifulSoup(content, "html.parser")
            body = soup.find("body")
            text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info("Saved: %s", filepath)
            return filepath
    except Exception as e:
        logger.warning("Failed to fetch %s : %s", url, e)
        return None

async def fetch_web_pages(queries: list[str], num_results: int, provider: str, download_dir: str = "./downloaded"):
    os.makedirs(download_dir, exist_ok=True)

    async with aiohttp.ClientSession(headers=REQUESTS_HEADER) as session:
        for query in queries:
            urls = get_urls(query, num_results, provider)
            if not urls:
                logger.info("No URLs returned for query: %s", query)
                continue

            for url in urls:
                if not (url.startswith("http://") or url.startswith("https://")):
                    logger.debug("Skipping non-http url: %s", url)
                    continue
                result = await fetch_and_save(session, url, download_dir)
                
                if result is None:
                    logger.debug("Sleeping 2s after fetch failure to avoid quick re-block")
                    await asyncio.sleep(2.0)
                else:
                    await asyncio.sleep(1.0)

            await asyncio.sleep(1.0)

    return True
def remove_temp_files(download_dir: str = "./downloaded"):
    try:
        if not os.path.exists(download_dir):
            return
        for filename in os.listdir(download_dir):
            file_path = os.path.join(download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        logger.warning("Failed to remove temp files: %s", e)