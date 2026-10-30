#!/usr/bin/env python3
import json
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup


def collect_wikipedia_articles(
    num_articles=50, min_size=2000, days_ago=90, save_path="wikipedia_articles.json"
):
    """
    Collect recently created Wikipedia articles and save them to a JSON file.
    Includes HTML, plain text, and wikitext (source) formats.

    Parameters:
    - num_articles: Number of articles to collect
    - min_size: Minimum article size in bytes
    - days_ago: How far back to look for articles
    - save_path: Where to save the JSON file
    """
    # API endpoint and headers
    API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "WikipediaArticleCollector/0.1 (your-email@example.com)"}

    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)

    # Format dates for API
    rcstart = end_date.strftime("%Y%m%d%H%M%S")
    rcend = start_date.strftime("%Y%m%d%H%M%S")

    # Initialize variables
    all_articles = []
    continue_param = None
    batch_size = 50  # API max is usually 50

    print(f"Starting article collection from {start_date.date()} to {end_date.date()}")

    # Collect articles with pagination
    while len(all_articles) < 500:  # Cap at 500 to avoid too many requests
        # Set up parameters
        params = {
            "action": "query",
            "format": "json",
            "list": "recentchanges",
            "rctype": "new",
            "rcnamespace": "0",  # Main article namespace
            "rclimit": batch_size,
            "rcprop": "title|timestamp|ids|sizes|user",
            "rcshow": "!redirect",
            "rcstart": rcstart,
            "rcend": rcend,
            "rcdir": "older",
        }

        # Add continue parameter if we have one
        if continue_param:
            params.update(continue_param)

        # Make the request
        response = requests.get(API_ENDPOINT, headers=HEADERS, params=params)
        data = response.json()

        # Extract articles
        if "query" in data and "recentchanges" in data["query"]:
            batch = data["query"]["recentchanges"]
            all_articles.extend(batch)
            print(f"Retrieved {len(batch)} articles, total: {len(all_articles)}")

            # Check if we need to continue
            if "continue" in data:
                continue_param = data["continue"]
            else:
                break
        else:
            print("No more results or error in API response.")
            break

        # Be nice to the API
        time.sleep(1)

    # Filter articles by size
    filtered_articles = [
        article for article in all_articles if article.get("newlen", 0) >= min_size
    ]
    print(f"Articles after size filtering: {len(filtered_articles)}")

    # Function to get article wikitext (source)
    def get_article_wikitext(title):
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "formatversion": "2",
        }

        response = requests.get(API_ENDPOINT, headers=HEADERS, params=params)
        return response.json()

    # Function to get article HTML and metadata
    def get_article_html(title):
        params = {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "text|sections|categories|links|templates|externallinks",
            "formatversion": "2",
        }

        response = requests.get(API_ENDPOINT, headers=HEADERS, params=params)
        return response.json()

    # Sort by size and take the top articles
    filtered_articles.sort(key=lambda x: x.get("newlen", 0), reverse=True)
    selected_articles = filtered_articles[:num_articles]

    # Collect detailed information for each article
    detailed_articles = []

    for index, article in enumerate(selected_articles, 1):
        title = article["title"]
        print(f"Processing article {index}/{len(selected_articles)}: {title}")

        # Get wikitext (source markup)
        wikitext_data = get_article_wikitext(title)

        # Get HTML and metadata
        html_data = get_article_html(title)

        # Skip if we couldn't get the article
        if "query" not in wikitext_data or "parse" not in html_data:
            print(f"Could not retrieve content for {title}, skipping...")
            continue

        # Extract wikitext
        pages = wikitext_data["query"]["pages"]
        if len(pages) > 0 and "revisions" in pages[0]:
            wikitext = pages[0]["revisions"][0]["slots"]["main"]["content"]
        else:
            wikitext = "Error: Could not retrieve wikitext"

        # Extract HTML and metadata
        html_content = html_data["parse"]["text"]
        sections = html_data["parse"].get("sections", [])
        categories = html_data["parse"].get("categories", [])
        external_links = html_data["parse"].get("externallinks", [])

        # Parse HTML to get plain text
        soup = BeautifulSoup(html_content, "html.parser")
        plain_text = soup.get_text(separator="\n")

        # Create article dictionary
        article_info = {
            "title": title,
            "page_id": html_data["parse"].get("pageid"),
            "timestamp": article["timestamp"],
            "size": article.get("newlen", 0),
            "num_sections": len(sections),
            "section_titles": [section.get("line", "") for section in sections],
            "num_external_links": len(external_links),
            "external_links": external_links,
            "categories": [cat.get("*", "") for cat in categories],
            "plain_text": plain_text,
            "html_content": html_content,
            "wikitext": wikitext,
        }

        detailed_articles.append(article_info)

        # Save individual wikitext file
        wikitext_filename = f"{title.replace(' ', '_').replace('/', '_')}.wiki"
        with open(wikitext_filename, "w", encoding="utf-8") as wiki_file:
            wiki_file.write(wikitext)
        print(f"Saved wikitext file: {wikitext_filename}")

        # Be nice to the API
        time.sleep(1)

    # Save to JSON file
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(detailed_articles, json_file, indent=2, ensure_ascii=False)

    print(
        f"\nCollection complete! {len(detailed_articles)} articles saved to {save_path}"
    )
    return detailed_articles


# Example usage
if __name__ == "__main__":
    # Collect 10 articles and save to 'wikipedia_articles.json'
    articles = collect_wikipedia_articles(num_articles=10, min_size=5000, days_ago=30)

    # Print titles of collected articles
    print("\nCollected articles:")
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['title']} ({article['size']} bytes)")
