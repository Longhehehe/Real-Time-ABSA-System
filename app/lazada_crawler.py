"""
Lazada Crawler Module (Selenium-based)
Full-featured crawler with:
- Balanced mode (1-3 stars priority)
- Anti-bot detection
- Debug mode
- Search functionality

Upgraded from request-based to Selenium for better reliability.
"""
import time
import random
import re
import json
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


class LazadaCrawler:
    """Selenium-based Lazada Review Crawler with anti-bot detection."""
    
    # Random User Agents để giả lập nhiều trình duyệt khác nhau
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    ]
    
    def __init__(self, cookie_list=None, keep_alive=False, debug_mode=False):
        """
        Initializes the crawler with optional cookies.
        
        Args:
            cookie_list (list): List of dicts [{'name': '...', 'value': '...'}, ...]
            keep_alive (bool): If True, driver is not closed automatically.
            debug_mode (bool): If True, saves screenshots at key points for debugging.
        """
        chrome_options = Options()
        
        # === ANTI-BOT DETECTION ===
        # Random User Agent
        user_agent = random.choice(self.USER_AGENTS)
        chrome_options.add_argument(f"--user-agent={user_agent}")
        
        # Disable automation flags
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Other stealth options
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--lang=vi-VN,vi")  # Vietnamese language
        
        if keep_alive:
            chrome_options.add_experimental_option("detach", True)
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # Remove webdriver property (anti-detection)
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['vi-VN', 'vi', 'en-US', 'en']
                });
                window.chrome = { runtime: {} };
            """
        })
        
        self.wait = WebDriverWait(self.driver, 10)
        self.cookie_list = cookie_list
        self.keep_alive = keep_alive
        self.debug_mode = debug_mode
        self.debug_counter = 0
        print(f"[STEALTH] Using User-Agent: {user_agent[:50]}...")

    def debug_screenshot(self, label):
        """Saves a screenshot if debug_mode is enabled."""
        if self.debug_mode:
            self.debug_counter += 1
            filename = f"debug_{self.debug_counter:03d}_{label}.png"
            try:
                self.driver.save_screenshot(filename)
                print(f"[DEBUG] Screenshot saved: {filename}")
            except Exception as e:
                print(f"[DEBUG] Failed to save screenshot: {e}")

    def open_login_page(self):
        print("Opening login page...")
        self.driver.get("https://member.lazada.vn/user/login.htm")

    def get_cookies(self):
        """Returns the current cookies from the driver."""
        return self.driver.get_cookies()

    def close(self):
        if self.driver:
            self.driver.quit()

    def load_cookies(self, url="https://www.lazada.vn"):
        # If cookies are already loaded/session is active, skip logic
        if self.driver.current_url != "data:,": # Check if driver has loaded something
             curr_domain = self.driver.current_url
             if "lazada" in curr_domain:
                 return # Already on Laz, assume session is good or manual login handled it

        print("Navigating to domain to set cookies...")
        self.driver.get(url)
        
        if self.cookie_list:
            print(f"Injecting {len(self.cookie_list)} cookies...")
            for cookie in self.cookie_list:
                try:
                    self.driver.add_cookie(cookie)
                except Exception as e:
                     pass
            
            self.driver.refresh()
            time.sleep(1) # Reduced from 3

    def search_product(self, keyword):
        """Search for products on Lazada."""
        print(f"Searching for: {keyword}")
        
        if self.cookie_list and "lazada" not in self.driver.current_url:
             self.load_cookies()
        elif "lazada" not in self.driver.current_url:
            self.driver.get("https://www.lazada.vn")
        
        try:
            search_box = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search']")))
            search_box.clear()
            search_box.send_keys(keyword)
            search_box.send_keys(Keys.RETURN)
            
            # SCROLL TO LOAD MORE PRODUCTS
            print("Scrolling to load more products...")
            for i in range(5): # Scroll 5 times
                self.driver.execute_script(f"window.scrollTo(0, {(i+1)*1000});")
                # Random wait
                time.sleep(random.uniform(0.8, 1.5))
            
            # Wait for results visibility
            try:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-qa-locator='product-item']")))
            except:
                time.sleep(random.uniform(2, 3)) # Fallback wait
            
            # Scrape results
            products = []
            
            # Try multiple generic selectors for product items
            items = self.driver.find_elements(By.CSS_SELECTOR, "div[data-qa-locator='product-item']")
            if not items:
                items = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='GridItem']")
            
            if not items:
                 # Try finding via links directly
                 links = self.driver.find_elements(By.XPATH, "//a[contains(@href, '/products/') and string-length(text()) > 5]")
                 seen_links = set()
                 unique_links = []
                 for l in links:
                     href = l.get_attribute('href')
                     if href and href not in seen_links:
                         unique_links.append(l)
                         seen_links.add(href)
                 
                 for link_el in unique_links[:50]: # Limit to 50
                      try:
                          products.append({
                              "title": link_el.text,
                              "price": "See details",
                              "link": link_el.get_attribute('href'),
                              "reviews": 0
                          })
                      except:
                          pass
                 return products

            print(f"Found {len(items)} items. Parsing top 50...")
            for item in items[:50]: # Increased limit from 10 to 50
                try:
                    # Title
                    try:
                        title_el = item.find_element(By.CSS_SELECTOR, "a[title]")
                        title = title_el.get_attribute("title")
                    except:
                        try:
                            title_el = item.find_element(By.CSS_SELECTOR, "a")
                            title = title_el.text
                        except:
                            title = "Unknown Product"
                            
                    # Link
                    try:
                        link = item.find_element(By.TAG_NAME, "a").get_attribute("href")
                    except:
                        link = "#"

                    # Price
                    try:
                         price = item.find_element(By.CSS_SELECTOR, "span[class*='price']").text
                    except:
                         price = "N/A"
                         
                    # Review Count
                    review_count = 0
                    try:
                        # Often looks like "(123)" or "123 Reviews"
                        # Selectors: .c3XbGJ (rating count), or span containing '('
                        review_text_el = item.find_element(By.CSS_SELECTOR, "span[class*='rating__review']") 
                        review_text = review_text_el.text.replace("(", "").replace(")", "").replace(".", "").replace(",", "")
                        review_count = int(review_text)
                    except:
                        try:
                             # Fallback: look for text with parentheses inside the item
                             # item content usually has structure.
                             spans = item.find_elements(By.TAG_NAME, "span")
                             for s in spans:
                                 t = s.text.strip()
                                 if t.startswith("(") and t.endswith(")"):
                                     review_text = t[1:-1].replace(".", "").replace(",", "")
                                     if review_text.isdigit():
                                         review_count = int(review_text)
                                         break
                        except:
                             pass
                        
                    products.append({
                        "title": title,
                        "price": price,
                        "link": link,
                        "reviews": review_count
                    })
                except Exception as e:
                    print(f"Error parsing item: {e}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def crawl_reviews(self, product_url, max_pages=5, balanced_mode=False):
        """
        Crawl reviews from a Lazada product page.
        
        Args:
            product_url: URL of the product
            max_pages: Max pages per star rating (-1 for unlimited)
            balanced_mode: If True, prioritizes 1-3 star reviews first
            
        Returns:
            pandas DataFrame with reviews
        """
        print(f"Crawling reviews for: {product_url} (Balanced: {balanced_mode})")
        if not product_url.startswith("http"):
            product_url = "https:" + product_url
            
        self.driver.get(product_url)
        
        # WAIT 3 SECONDS for page to fully load
        print("Waiting 3 seconds for page to fully load...")
        time.sleep(3)
        
        # SCROLL DOWN GRADUALLY until review module is found
        print("Scrolling down to find review section...")
        found_review_module = False
        
        for scroll_attempt in range(30):  # Max 30 attempts (~ 9000px down)
            # Scroll down by 300px each time
            self.driver.execute_script("window.scrollBy(0, 400);")
            time.sleep(0.3)
            
            # Check if review module is visible
            try:
                review_module = self.driver.find_element(By.ID, "module_product_review")
                if review_module.is_displayed():
                    print(f"Review section found after {scroll_attempt + 1} scrolls!")
                    # Center it in view
                    self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", review_module)
                    time.sleep(1)
                    found_review_module = True
                    break
            except:
                pass
        
        if not found_review_module:
            print("Could not find review section after scrolling. Continuing anyway...")
        else:
            # WAIT for reviews/filter to actually load (content is lazy-loaded)
            print("Waiting for review content to load...")
            try:
                self.wait.until(EC.presence_of_element_located((By.XPATH, "//div[@id='module_product_review']//*[contains(text(), 'Filter') or contains(text(), 'star') or contains(text(), 'Sao')]")))
                time.sleep(3)  # Wait 3 seconds for reviews to fully load
                print("Review content loaded!")
            except:
                print("Review content didn't load in time, continuing anyway...")
                time.sleep(3)  # Fallback extra wait

        all_reviews = []
        
        # Define groups
        if balanced_mode:
            crawl_order = [1, 2, 3, 4, 5]
            low_star_count = 0
            high_star_count = 0
        else:
            crawl_order = [5, 4, 3, 2, 1]
        
        first_star_done = False  # Track if first star has been processed
        
        for star in crawl_order:
            print(f"\n{'='*50}")
            print(f">>> STARTING iteration for {star} Star <<<")
            print(f"{'='*50}")
            self.debug_screenshot(f"start_star_{star}")
            
            # Check limits for balanced mode
            # PRIORITY: Exhaust all 1,2,3 star reviews first, then add 4,5 stars to match
            if balanced_mode:
                if star in [4, 5]:
                    # Only skip if we already have enough high stars to match low stars
                    if low_star_count == 0:
                        print(f"Skipping {star} Star: No low star reviews collected yet")
                        continue
                    if high_star_count >= low_star_count:
                        print(f"Skipping {star} Star: Already balanced (high={high_star_count}, low={low_star_count})")
                        continue
                    print(f"Crawling {star} Star: Need more to balance (high={high_star_count}, target={low_star_count})")

            try:
                # SCROLL BACK UP to filter - ONLY if we've already processed at least one star
                # On the first star, we're already at the review section from initial scroll
                if first_star_done:
                    print(f"Returning to review module for {star} Star filter...")
                    
                    # DIRECT SCROLL to review module top (don't use incremental scroll which overshoots)
                    try:
                        review_module = self.driver.find_element(By.ID, "module_product_review")
                        self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'start'});", review_module)
                        time.sleep(2)
                        print("Scrolled to review module top.")
                    except:
                        print("Could not find review module. Scrolling up slightly...")
                        self.driver.execute_script("window.scrollBy(0, -500);")
                        time.sleep(1)
                    
                # 1. Click "Filter by" dropdown
                # CORRECT SELECTOR from browser inspection: .pdp-mod-filterSort-v2 .oper (first one is Filter)
                try:
                    # Find filter buttons - there are 2: "Filter by" and "Sort by"
                    filter_opers = self.driver.find_elements(By.CSS_SELECTOR, ".pdp-mod-filterSort-v2 .oper")
                    filter_btn = None
                    
                    for oper in filter_opers:
                        if "Filter" in oper.text or "Lọc" in oper.text or "star" in oper.text.lower() or "sao" in oper.text.lower():
                            filter_btn = oper
                            break
                    
                    if filter_btn:
                        print(f"Found filter button: '{filter_btn.text[:30]}...'")
                        self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", filter_btn)
                        time.sleep(0.5)
                        filter_btn.click()
                        time.sleep(1.5)  # Wait for dropdown to appear
                        self.debug_screenshot(f"after_filter_click_star_{star}")
                    else:
                        print("Filter button not found with CSS selector. Trying XPath fallback...")
                        # Fallback to old XPath method
                        filter_triggers = self.driver.find_elements(By.XPATH, "//*[(contains(text(), 'Filter') or contains(text(), 'All stars') or contains(text(), 'Tất cả'))]")
                        for trig in filter_triggers:
                            if trig.is_displayed():
                                trig.click()
                                time.sleep(1)
                                break
                except Exception as e:
                    print(f"Error clicking filter: {e}")
                    self.debug_screenshot(f"filter_error_star_{star}")

                # 2. Click specific star rating from dropdown
                # CORRECT SELECTOR from browser inspection: .next-menu-item (li elements in dropdown)
                filter_found = False
                for attempt in range(2): # Retry logic
                    try:
                        # Wait for dropdown to fully render
                        time.sleep(1)
                        
                        # Use the correct CSS selector: .next-menu-item
                        star_options = self.driver.find_elements(By.CSS_SELECTOR, ".next-menu-item")
                        print(f"Found {len(star_options)} menu items in dropdown")
                        
                        # Star text formats: "1 star", "2 star", etc. (English) or "1 sao", "2 sao" (Vietnamese)
                        target_texts = [f"{star} star", f"{star} sao", f"{star} Star", f"{star} Sao"]
                        
                        if not star_options:
                            # Dropdown may not have opened, try re-clicking filter
                            if attempt == 0:
                                print(f"No dropdown items found. Re-opening filter menu...")
                                try:
                                    filter_opers = self.driver.find_elements(By.CSS_SELECTOR, ".pdp-mod-filterSort-v2 .oper")
                                    for oper in filter_opers:
                                        if "Filter" in oper.text or "star" in oper.text.lower() or "sao" in oper.text.lower():
                                            oper.click()
                                            time.sleep(1.5)
                                            break
                                except:
                                    pass
                                continue
                            else:
                                print(f"Filter dropdown still empty after retry. Skipping star {star}.")
                                break
    
                        clicked_star = False
                        for opt in star_options:
                            try:
                                opt_text = opt.text.strip() if opt.text else ""
                                print(f"  Menu item: '{opt_text}'")
                                
                                # Check if this option matches our target star
                                if opt_text in target_texts or any(t.lower() == opt_text.lower() for t in target_texts):
                                    print(f"  >>> MATCHED target: {star} star <<<")
                                    
                                    # Use JavaScript click for reliability (dropdown elements can be tricky)
                                    self.driver.execute_script("arguments[0].click();", opt)
                                    
                                    clicked_star = True
                                    filter_found = True
                                    
                                    # Wait for the review list to refresh
                                    print(f"Clicked {star} Star. Waiting for reviews to reload...")
                                    time.sleep(1.5)
                                    
                                    try:
                                        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='item'][class*='review']")))
                                    except:
                                        print("Timed out waiting for reviews to load. (Might be 0 reviews)")
                                        
                                    time.sleep(random.uniform(1, 2))
                                    break
                            except Exception as click_err:
                                print(f"  Click attempt failed: {click_err}")
                                continue
                        
                        if clicked_star:
                            break
                        
                        if not clicked_star and attempt == 1:
                             print(f"Star {star} option not found in dropdown. Skipping.")
                             self.driver.save_screenshot(f"debug_star_{star}_click_fail.png")
                             
                    except Exception as e:
                         print(f"Error clicking star {star}: {e}")
                         self.driver.save_screenshot(f"debug_star_{star}_error.png")
                
                if not filter_found:
                    continue

                # Pagination Loop
                page = 1
                while True:
                    if max_pages != -1 and page > max_pages:
                        break
                        
                    if balanced_mode and star in [4, 5]:
                        remaining_quota = low_star_count - high_star_count
                        if remaining_quota <= 0:
                            break
                        
                    print(f"Page {page} of {star} stars")
                    # Wait for content load
                    time.sleep(random.uniform(1, 2))
                    
                    # Extract visible reviews - IMPROVED SELECTORS
                    # 1. Look for container first
                    container = None
                    try:
                        container = self.driver.find_element(By.CSS_SELECTOR, ".mod-reviews")
                    except:
                        pass
                    
                    if container:
                        review_items = container.find_elements(By.CSS_SELECTOR, ".item")
                    else:
                        # Fallback global search
                        review_items = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='item-content']")
                    
                    if not review_items:
                         print(f"No review items found on page {page} for star {star}")

                    for item in review_items:
                        if balanced_mode and star in [4, 5] and high_star_count >= low_star_count:
                             break

                        try:
                            # Extract review content
                            try:
                                content = item.find_element(By.CSS_SELECTOR, "div[class*='content']").text
                            except:
                                try:
                                    content = item.find_element(By.CSS_SELECTOR, ".item-content").text
                                except:
                                    content = ""
                            
                            # Extract author/buyer name
                            try:
                                author = item.find_element(By.CSS_SELECTOR, "div[class*='middle'] span").text
                            except:
                                try:
                                    author = item.find_element(By.CSS_SELECTOR, "span[class*='name']").text
                                except:
                                    author = "Anonymous"
                            
                            # Extract review time
                            try:
                                review_time = item.find_element(By.CSS_SELECTOR, "span[class*='date'], span[class*='time']").text
                            except:
                                review_time = ""
                            
                            # Extract rating (star count)
                            rating = star  # Default from filter
                            try:
                                star_container = item.find_element(By.CSS_SELECTOR, "div[class*='star'], span[class*='star']")
                                # Try to extract from class like "star-5" or "rate-5"
                                star_class = star_container.get_attribute("class") or ""
                                for i in range(5, 0, -1):
                                    if str(i) in star_class:
                                        rating = i
                                        break
                            except:
                                pass
                            
                            # Extract like count
                            like_count = 0
                            try:
                                like_elem = item.find_element(By.CSS_SELECTOR, "span[class*='like'], div[class*='like']")
                                like_text = like_elem.text
                                # Extract number from text like "12 likes" or "(12)"
                                numbers = re.findall(r'\d+', like_text)
                                if numbers:
                                    like_count = int(numbers[0])
                            except:
                                pass
                            
                            # Extract SKU info (variant info)
                            sku_info = ""
                            try:
                                sku_info = item.find_element(By.CSS_SELECTOR, "div[class*='sku'], span[class*='sku'], div[class*='variant']").text
                            except:
                                pass
                            
                            # Extract images
                            images = []
                            try:
                                img_elements = item.find_elements(By.CSS_SELECTOR, "img[src*='lazada'], img[src*='slatic']")
                                images = [img.get_attribute("src") for img in img_elements if img.get_attribute("src")]
                            except:
                                pass

                            # Detect if review has actual text content (not just SKU info)
                            # Empty reviews look like: "Nhóm màu: X\nKích thước: Y\nHelpful(0)"
                            has_text_content = True
                            if content:
                                # Remove SKU patterns and Helpful(N) to check for actual content
                                cleaned = content
                                # Remove patterns like "Nhóm màu: X", "Size: Y", "Kích thước: Z"
                                cleaned = re.sub(r'(Nhóm\s*[Mm]àu|Size|Kích\s*thước|Color|Màu\s*sắc)\s*:\s*[^\n]+', '', cleaned)
                                # Remove Helpful(N) pattern
                                cleaned = re.sub(r'Helpful\s*\(\d+\)', '', cleaned)
                                # Remove Material: pattern
                                cleaned = re.sub(r'Material\s*:\s*[^\n]+', '', cleaned)
                                # Remove remaining whitespace and newlines
                                cleaned = cleaned.strip()
                                # If nothing left after cleaning, it's an empty review
                                has_text_content = len(cleaned) > 0
                            else:
                                has_text_content = False

                            all_reviews.append({
                                "reviewContent": content,
                                "rating": rating,
                                "buyerName": author,
                                "reviewTime": review_time,
                                "likeCount": like_count,
                                "skuInfo": sku_info,
                                "images": str(images) if images else "",
                                "star_filter": star,
                                "has_text_content": has_text_content  # True if review has actual text
                            })
                            
                            if balanced_mode and has_text_content:  # Only count reviews WITH actual text
                                if star in [1, 2, 3]: low_star_count += 1
                                elif star in [4, 5]: high_star_count += 1
                                
                        except:
                            continue
                    
                    if balanced_mode and star in [4, 5] and high_star_count >= low_star_count:
                        print(f"✓ Balanced reached for {star} Star! high={high_star_count}, low={low_star_count}. Moving to next star...")
                        break

                    # Go to next page by clicking page numbers (2, 3, 4...)
                    reached_last_page = False  # Flag to track if we should exit pagination
                    try:
                        # CRITICAL: Scroll to pagination area by finding "Page x out of Y" text
                        try:
                            page_info = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Page') and contains(text(), 'out of')]")
                            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", page_info)
                            time.sleep(1)
                            
                            # Parse "Page X out of Y" to get total pages (Y)
                            page_text = page_info.text
                            match = re.search(r'Page\s*(\d+)\s*out of\s*(\d+)', page_text)
                            if match:
                                total_pages = int(match.group(2))
                                print(f"Total pages: {total_pages}, Current page variable: {page}")
                                
                                # Check if we've clicked all pages (page variable is 1-based)
                                if page >= total_pages:
                                    print(f"=== COMPLETED all pages for {star} Star ({page}/{total_pages}). Will move to next star. ===")
                                    self.debug_screenshot(f"completed_star_{star}_page_{page}_of_{total_pages}")
                                    reached_last_page = True
                        except Exception as e:
                            print(f"Could not parse pagination: {e}")
                            # Fallback: Scroll to bottom of review module
                            try:
                                module = self.driver.find_element(By.ID, 'module_product_review')
                                self.driver.execute_script("arguments[0].scrollIntoView(false);", module)
                                time.sleep(1)
                            except:
                                pass
                        
                        # If we reached last page, break out of pagination loop
                        if reached_last_page:
                            break

                        # Click NEXT page number (page + 1)
                        next_page_num = page + 1
                        print(f"Looking for page number: {next_page_num}")
                        
                        page_num_btn = None
                        try:
                            page_num_btn = self.driver.find_element(By.XPATH, f"//div[@id='module_product_review']//*[normalize-space(text())='{next_page_num}']")
                        except:
                            print(f"Page number {next_page_num} not found in DOM. End of pagination.")
                            break
                        
                        if not page_num_btn:
                            print(f"Page number {next_page_num} element not found.")
                            break
                        
                        # Check if disabled
                        btn_class = (page_num_btn.get_attribute("class") or "").lower()
                        if "disabled" in btn_class:
                            print(f"Page {next_page_num} is disabled. End of pages.")
                            break
                        
                        # Scroll to and click the page number
                        self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", page_num_btn)
                        time.sleep(random.uniform(0.5, 1))
                        
                        print(f"Clicking page number {next_page_num}...")
                        page_num_btn.click()
                        
                        # Wait for page update - simplified check
                        try:
                             self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='item'][class*='review']")))
                        except:
                             print("Wait for page load timed out (or empty page).")
                             
                        time.sleep(random.uniform(1, 2))
                        
                        # Base wait (reduced for speed)
                        sleep_time = random.uniform(1.5, 3)
                        
                        # Occasional "Long Pause" - 5% chance (reduced from 15%)
                        if random.random() < 0.05:
                             long_pause = random.uniform(2, 4)
                             print(f"Taking a short break... ({long_pause:.1f}s)")
                             sleep_time += long_pause
                             
                        time.sleep(sleep_time)
                        page += 1
                    except:
                        print("Next button not found (end of pages).")
                        break 
                        
            except Exception as e:
                print(f"Error processing star {star}: {e}")
                self.debug_screenshot(f"error_star_{star}")
            
            # Log completion of this star rating
            print(f"=== FINISHED processing {star} Star. Total reviews collected so far: {len(all_reviews)} ===")
            
            # Mark first star as done so subsequent stars will scroll up first
            first_star_done = True
        
        if not self.keep_alive:        
            self.driver.quit()
            
        if not all_reviews:
            print("No reviews found. Saving debug info...")
            timestamp = int(time.time())
            try:
                self.driver.save_screenshot(f"debug_crawl_error_{timestamp}.png")
                with open(f"debug_crawl_error_{timestamp}.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
            except:
                pass
        
        return pd.DataFrame(all_reviews)

    def extract_visible_reviews(self, override_star=None):
        """
        Extracts reviews currently visible on the page without navigation or filtering.
        Useful for 'Assisted Mode'.
        """
        reviews = []
        try:
            # Similar selector to crawl_reviews
            # Look for review items
            review_items = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='item'][class*='review']")
            if not review_items:
                review_items = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='review-item']")
            
            print(f"Found {len(review_items)} visible reviews.")
            
            for item in review_items:
                try:
                    content = item.find_element(By.CSS_SELECTOR, "div[class*='content']").text
                except:
                    content = ""
                
                try:
                    author = item.find_element(By.CSS_SELECTOR, "div[class*='middle'] span").text
                except:
                    author = "Anonymous"
                
                star_rating = override_star if override_star else "Manual/Unknown"

                reviews.append({
                    "star": star_rating,
                    "author": author,
                    "content": content
                })
        except Exception as e:
            print(f"Error extracting visible reviews: {e}")
            
        return pd.DataFrame(reviews)


# ============================================
# WRAPPER FUNCTIONS for backward compatibility
# with old request-based interface
# ============================================

def extract_item_id(url: str) -> Optional[str]:
    """
    Extract product item_id from Lazada URL.
    Supports multiple URL formats.
    """
    if not url:
        return None
    
    patterns = [
        r'-i(\d+)-s',           # Format: -i123456-s
        r'-i(\d+)\.',           # Format: -i123456.html
        r'-i(\d+)$',            # Format: ends with -i123456
        r'-i(\d+)\?',           # Format: -i123456?
        r'itemId=(\d+)',        # Query param: itemId=123456
        r'/i(\d+)\?',           # Format: /i123456?
        r'/i(\d+)$',            # Format: /i123456
        r'products/.*?-(\d+)\.html',  # Format: products/xxx-123456.html
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Fallback: find any 6+ digit number in URL
    all_numbers = re.findall(r'(\d{6,})', url)
    if all_numbers:
        return all_numbers[0]
    
    return None


def crawl_reviews(
    product_url: str,
    cookies_path: Optional[str] = None,
    max_reviews: int = 100,
    delay_min: float = 2.0,
    delay_max: float = 5.0,
    item_id: Optional[str] = None,
    balanced_mode: bool = False
) -> Tuple[List[Dict], str]:
    """
    Wrapper function that maintains old interface but uses Selenium crawler.
    
    Args:
        product_url: Lazada product URL
        cookies_path: Path to cookies JSON file (optional)
        max_reviews: Maximum reviews to collect
        delay_min/delay_max: Not used in Selenium version (kept for compatibility)
        item_id: Not used (extracted from URL)
        balanced_mode: If True, prioritize 1-3 star reviews
        
    Returns:
        Tuple of (list of reviews as dicts, error message)
    """
    try:
        # Load cookies from JSON if provided
        cookie_list = None
        if cookies_path:
            json_path = cookies_path
            if cookies_path.endswith('.txt'):
                json_path = cookies_path.replace('.txt', '.json')
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        cookie_list = json.load(f)
                    print(f"✅ Loaded {len(cookie_list)} cookies from {json_path}")
                except Exception as e:
                    print(f"⚠️ Could not load cookies: {e}")
        
        # Calculate max pages based on max_reviews (approx 10 reviews per page)
        max_pages = max(1, max_reviews // 10)
        
        # Create crawler and run
        crawler = LazadaCrawler(cookie_list=cookie_list, keep_alive=False, debug_mode=False)
        df = crawler.crawl_reviews(product_url, max_pages=max_pages, balanced_mode=balanced_mode)
        
        if df.empty:
            return [], "Không lấy được review nào. Có thể Lazada chặn bot."
        
        # Convert DataFrame to list of dicts
        reviews = df.to_dict('records')
        
        # Limit to max_reviews
        reviews = reviews[:max_reviews]
        
        print(f"✅ Đã crawl {len(reviews)} reviews")
        return reviews, ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [], f"Lỗi: {str(e)}"


def reviews_to_dataframe(reviews: List[Dict]) -> pd.DataFrame:
    """
    Convert list of reviews to pandas DataFrame.
    """
    if not reviews:
        return pd.DataFrame()
    return pd.DataFrame(reviews)


# Test
if __name__ == "__main__":
    print("=== Lazada Selenium Crawler Test ===")
    crawler = LazadaCrawler(keep_alive=True, debug_mode=True)
    # crawler.search_product("iphone")
