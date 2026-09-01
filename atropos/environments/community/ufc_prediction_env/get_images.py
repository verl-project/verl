import os
import random
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

# 1. Load your CSV
df = pd.read_csv("fighter_stats.csv")

# List of different user agents to rotate between
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/92.0.4515.159 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36",
]

# List of different accept languages
ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-CA,en;q=0.9",
    "en-AU,en;q=0.9",
    "en-NZ,en;q=0.9",
]


def get_random_headers():
    """Generate a random set of headers for each request."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": random.choice(ACCEPT_LANGUAGES),
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1",
    }


# 2. Generate UFC profile URLs
def make_url(name):
    """Turn a fighter name into a UFC athlete URL slug."""
    if not isinstance(name, str) or not name.strip():
        return None
    slug = name.lower().strip()
    slug = re.sub(r"[''']", "", slug)  # Remove apostrophes
    slug = re.sub(r"[^a-z\s-]", "", slug)  # Keep letters, spaces, hyphens
    slug = re.sub(r"\s+", "-", slug)  # Spaces → hyphens
    return f"https://www.ufc.com/athlete/{slug}"


df["UFC_Profile_URL"] = df["name"].apply(make_url)

# 3. Prepare output folder
output_folder = "fighter_images"
os.makedirs(output_folder, exist_ok=True)


# 4. Scrape & download each image
def download_ufc_image(url, counter):
    """Fetch the UFC athlete page, parse out the main profile image, and save it."""
    try:
        # Derive filename from URL slug
        slug = url.rstrip("/").split("/")[-1]
        filename = f"{slug}.jpg"
        path = os.path.join(output_folder, filename)

        # Check if image already exists
        if os.path.exists(path):
            print(f"[i] Image already exists for {filename}")
            return

        # Get fresh random headers for each request
        headers = get_random_headers()

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        img_tag = soup.select_one(
            "div.hero-profile__image-wrap img.hero-profile__image"
        )
        if not img_tag or not img_tag.get("src"):
            print(f"[!] No image found at {url}")
            return
        img_url = img_tag["src"]

        # Get fresh random headers for image download
        headers = get_random_headers()
        img_data = requests.get(img_url, headers=headers).content
        with open(path, "wb") as f:
            f.write(img_data)
        print(f"[✓] Saved {filename}")

        # Add random delay between 1-3 seconds
        time.sleep(random.uniform(1, 3))

        # After every 100 downloads, take a longer break
        if counter % 100 == 0:
            print(f"[i] Taking a 30-second break after {counter} downloads...")
            time.sleep(30)

    except Exception as e:
        print(f"[✗] Error at {url}: {e}")


# 5. Iterate and download
counter = 0
for url in df["UFC_Profile_URL"].dropna().unique():
    counter += 1
    download_ufc_image(url, counter)
