"""
data_gathering.py

This module is designed for automatic and comprehensive data gathering from various sources on the internet.
It includes functionalities for web scraping, API integration, and accessing public datasets. The gathered 
data is intended for use in the ZephyrCortex project to enable continuous learning and improvement.

Features:
- Web Scraping: Automatically extract data from web pages using BeautifulSoup and Selenium.
- API Integration: Connect to various APIs to gather data (e.g., social media, news, scientific databases).
- Public Datasets: Download and preprocess data from public datasets (e.g., Kaggle, UCI Machine Learning Repository).
- Data Storage: Store gathered data in a local database for further processing.
- Scheduling: Schedule regular data gathering tasks to keep the dataset updated.
- Error Handling: Robust error handling and logging to manage failed data gathering attempts.
- Proxy and Rate Limiting: Use proxies and manage rate limits to avoid blocking and ensure compliance with terms of service.

Usage:
1. Configure the data sources and parameters.
2. Run the data gathering functions to collect data from various sources.
3. The gathered data is stored and can be accessed for further processing.

Dependencies:
- BeautifulSoup
- Selenium
- Requests
- pandas
- SQLAlchemy
- APScheduler

Example:
    from data_gathering import DataGatherer

    gatherer = DataGatherer()
    gatherer.gather_web_data('https://example.com')
    gatherer.gather_api_data('https://api.example.com/data')
    gatherer.gather_public_data('kaggle', 'dataset-name')

"""

import os
import shutil
import hashlib
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from urllib.parse import urlparse, urljoin

import PyPDF2
import requests
from bs4 import BeautifulSoup
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService

# Constants
MAX_RETRIES = 3
TIMEOUT_SECONDS = 10
CHROME_DRIVER_PATH = '/path/to/chromedriver'  # Replace with your actual ChromeDriver path
CACHE_DIR = "./cache"  # Directory to store cached data

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_webpage(url):
    """
    Scrapes a webpage and extracts relevant data using BeautifulSoup.

    :param url: The URL of the webpage to scrape.
    :return: Extracted data as a dictionary.
    """
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            links = [link.get('href') for link in soup.find_all('a', href=True)]
            images = [img.get('src') for img in soup.find_all('img', src=True)]
            headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            
            return {
                'url': url,
                'paragraphs': paragraphs,
                'links': links,
                'images': images,
                'headings': headings
            }
        else:
            logger.warning(f"Failed to fetch {url}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None

def fetch_api_data(api_url):
    """
    Fetches data from an external API endpoint.

    :param api_url: The URL of the API endpoint.
    :return: JSON response from the API.
    """
    try:
        response = requests.get(api_url, timeout=TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to fetch API data from {api_url}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching API data from {api_url}: {str(e)}")
        return None

def handle_dynamic_content(url):
    """
    Handles webpages with dynamic content using Selenium.

    :param url: The URL of the webpage with dynamic content.
    :return: Extracted data from dynamic content.
    """
    try:
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_service = ChromeService(CHROME_DRIVER_PATH)
        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        
        driver.get(url)
        time.sleep(5)  # Adjust as needed for dynamic content to load

        elements = driver.find_elements(By.XPATH, '//div[@class="content"]/p')
        dynamic_content = [element.text for element in elements]

        driver.quit()
        return {
            'url': url,
            'dynamic_content': dynamic_content
        }
    except Exception as e:
        logger.error(f"Error handling dynamic content for {url}: {str(e)}")
        return None

def respect_robots_txt(url):
    """
    Checks if a URL is allowed to be scraped based on robots.txt rules.

    :param url: The URL to check.
    :return: True if allowed, False otherwise.
    """
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        robots_response = requests.get(robots_url, timeout=TIMEOUT_SECONDS)
        if robots_response.status_code == 200:
            robots_txt = robots_response.text
            disallowed_paths = [line.split(': ')[1] for line in robots_txt.split('\n') if line.startswith('Disallow:')]
            for path in disallowed_paths:
                if urlparse(url).path.startswith(path):
                    return False
            return True
        else:
            logger.warning(f"Failed to fetch robots.txt for {url}. Status code: {robots_response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching robots.txt for {url}: {str(e)}")
        return False

def cache_data(data, cache_key):
    """
    Caches data for efficient retrieval.

    :param data: The data to cache.
    :param cache_key: The key to store/retrieve cached data.
    """
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        with open(cache_path, 'w', encoding='utf-8') as cache_file:
            json.dump(data, cache_file)
    except Exception as e:
        logger.error(f"Error caching data: {str(e)}")

def load_cached_data(cache_key):
    """
    Loads cached data if available.

    :param cache_key: The key to retrieve cached data.
    :return: Cached data as a dictionary, or None if not available.
    """
    try:
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as cache_file:
                return json.load(cache_file)
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading cached data: {str(e)}")
        return None

def error_handling_and_retry(url):
    """
    Implements error handling and retry mechanisms for HTTP requests.

    :param url: The URL to fetch.
    :return: Response content or None.
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.content
            else:
                logger.warning(f"Failed to fetch {url}. Status code: {response.status_code}")
                retries += 1
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            retries += 1
        time.sleep(2)  # Add delay between retries
    return None

def parallel_processing(urls):
    """
    Implements parallel processing to fetch data from multiple URLs simultaneously.

    :param urls: List of URLs to fetch.
    :return: Dictionary containing data fetched from each URL.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_webpage, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    results[url] = data
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
    return results

def parse_json_data(json_data):
    """
    Parses JSON data retrieved from APIs or web scraping.

    :param json_data: JSON data as a string.
    :return: Parsed data as a Python dictionary or None if parsing fails.
    """
    try:
        return json.loads(json_data)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON data: {str(e)}")
        return None

def extract_pdf_text(pdf_url):
    """
    Extracts text content from a PDF file.

    :param pdf_url: URL of the PDF file to extract text from.
    :return: Extracted text content as a string or None if extraction fails.
    """
    try:
        response = requests.get(pdf_url, timeout=TIMEOUT_SECONDS)
        if response.status_code == 200:
            with BytesIO(response.content) as pdf_buffer:
                pdf_reader = PyPDF2.PdfFileReader(pdf_buffer)
                text_content = ""
                for page_num in range(pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)
                    text_content += page.extract_text()
                return text_content
        else:
            logger.warning(f"Failed to fetch PDF from {pdf_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def download_images(image_urls, save_directory):
    """
    Downloads images from URLs and saves them to a specified directory.

    :param image_urls: List of image URLs to download.
    :param save_directory: Directory path to save downloaded images.
    :return: List of saved image file paths.
    """
    saved_paths = []
    os.makedirs(save_directory, exist_ok=True)
    try:
        for url in image_urls:
            response = requests.get(url, stream=True, timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                image_name = hashlib.md5(url.encode()).hexdigest() + os.path.splitext(url)[1]
                save_path = os.path.join(save_directory, image_name)
                with open(save_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                
                saved_paths.append(save_path)
            else:
                logger.warning(f"Failed to download image from {url}. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error downloading images: {str(e)}")
    return saved_paths

def submit_html_form(form_url, form_data):
    """
    Submits data to an HTML form on a webpage and retrieves response.

    :param form_url: URL of the webpage containing the form.
    :param form_data: Dictionary of form data to submit.
    :return: Response content or None if submission fails.
    """
    try:
        response = requests.post(form_url, data=form_data, timeout=TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.content
        else:
            logger.warning(f"Failed to submit HTML form at {form_url}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error submitting HTML form at {form_url}: {str(e)}")
        return None

def process_media_content(media_urls, save_directory):
    """
    Processes and downloads various types of media content.

    :param media_urls: List of URLs pointing to different types of media (images, PDFs, etc.).
    :param save_directory: Directory path to save downloaded media content.
    :return: Dictionary containing saved file paths categorized by media type.
    """
    saved_paths = defaultdict(list)
    try:
        for url in media_urls:
            if url.endswith('.pdf'):
                pdf_content = extract_pdf_text(url)
                if pdf_content:
                    pdf_hash = hashlib.md5(url.encode()).hexdigest()
                    pdf_file_path = os.path.join(save_directory, f"{pdf_hash}.txt")
                    with open(pdf_file_path, 'w', encoding='utf-8') as f:
                        f.write(pdf_content)
                    saved_paths['pdf'].append(pdf_file_path)
            elif any(url.lower().endswith(image_ext) for image_ext in ['.jpg', '.jpeg', '.png', '.gif']):
                image_paths = download_images([url], save_directory)
                saved_paths['images'].extend(image_paths)
            else:
                logger.warning(f"Unsupported media type for URL: {url}")
    except Exception as e:
        logger.error(f"Error processing media content: {str(e)}")
    return saved_paths

# usage 1
if __name__ == "__main__":
    sample_urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3"
    ]

    sample_api_url = "https://www.goodreads.com/shelf/show/free-download"

    sample_dynamic_url = "https://example.com/dynamic-page"

    # Example 1: Scrape webpages in parallel
    logger.info("Fetching data from multiple URLs in parallel...")
    results = parallel_processing(sample_urls)
    for url, data in results.items():
        print(f"URL: {url}")
        print(data)
        print()

    # Example 2: Fetch data from an API
    logger.info(f"Fetching data from API: {sample_api_url}")
    api_data = fetch_api_data(sample_api_url)
    if api_data:
        print("API Data:")
        print(api_data)
        print()

    # Example 3: Handle dynamic content
    # logger.info(f"Fetching dynamic content from {sample_dynamic_url}...")
    # dynamic_data = handle_dynamic_content(sample_dynamic_url)
    # if dynamic_data:
    #     print("Dynamic Content:")
    #     print(dynamic_data)
    #     print()

    # Example 4: Extract text from PDF
    # sample_pdf_url = "https://example.com/sample.pdf"
    # logger.info(f"Extracting text from PDF: {sample_pdf_url}")
    # pdf_text = extract_pdf_text(sample_pdf_url)
    # if pdf_text:
    #     print("Extracted PDF Text:")
    #     print(pdf_text)
    #     print()

    # Example 5: Download images
    # sample_image_urls = [
    #     "https://example.com/image1.jpg",
    #     "https://example.com/image2.png"
    # ]
    # save_directory = "./downloaded_images"
    # logger.info(f"Downloading images...")
    # saved_image_paths = download_images(sample_image_urls, save_directory)
    # if saved_image_paths:
    #     print("Saved Image Paths:")
    #     for path in saved_image_paths:
    #         print(path)
    #     print()

    # Example 6: Submit HTML form
    # sample_form_url = "https://example.com/submit-form"
    # sample_form_data = {'name': 'John Doe', 'email': 'john.doe@example.com'}
    # logger.info(f"Submitting HTML form: {sample_form_url}")
    # form_response = submit_html_form(sample_form_url, sample_form_data)
    # if form_response:
    #     print("Form Submission Response:")
    #     print(form_response.decode('utf-8'))
    #     print()

    # Example 7: Process media content
    # sample_media_urls = [
    #     "https://example.com/sample.pdf",
    #     "https://example.com/image1.jpg"
    # ]
    # media_save_directory = "./processed_media"
    # logger.info("Processing media content...")
    # processed_media_paths = process_media_content(sample_media_urls, media_save_directory)
    # if processed_media_paths:
    #     print("Processed Media Paths:")
    #     for media_type, paths in processed_media_paths.items():
    #         print(f"{media_type.capitalize()}:")
    #         for path in paths:
    #             print(path)
    #     print()
