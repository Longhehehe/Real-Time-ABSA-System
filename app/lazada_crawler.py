"""
Lazada Crawler Module
Crawl product reviews from Lazada using their API with cookies authentication.
"""
import os
import re
import json
import time
import random
import requests
import pandas as pd
from typing import Optional, Dict, List, Tuple
from fake_useragent import UserAgent
import http.cookiejar as cookielib


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


def extract_sku_id(url: str) -> Optional[str]:
    """
    Extract SKU ID from Lazada URL.
    Example: https://www.lazada.vn/products/pdp-i1216257-s1509400.html -> 1509400
    """
    match = re.search(r'-s(\d+)', url)
    return match.group(1) if match else None


def create_session(cookies_path: Optional[str] = None) -> requests.Session:
    """
    Create a requests session with browser-like headers.
    Optionally load cookies from file.
    """
    ua = UserAgent()
    session = requests.Session()
    
    # Browser-like headers
    session.headers.update({
        "User-Agent": ua.random,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.lazada.vn/",
        "Origin": "https://www.lazada.vn",
        "sec-ch-ua": '"Microsoft Edge";v="131", "Not=A?Brand";v="8"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    })
    
    # Load cookies if provided
    if cookies_path and os.path.exists(cookies_path):
        cookies_loaded = False
        
        # Try Netscape format first (.txt)
        if cookies_path.endswith('.txt'):
            try:
                cj = cookielib.MozillaCookieJar(cookies_path)
                cj.load(ignore_discard=True, ignore_expires=True)
                session.cookies.update(cj)
                print(f"✅ Loaded cookies from {cookies_path} (Netscape)")
                cookies_loaded = True
            except Exception as e:
                print(f"⚠️ Netscape format failed: {e}")
        
        # Try JSON format
        if not cookies_loaded:
            json_path = cookies_path.replace('.txt', '.json')
            if os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, 'r', encoding='utf-8') as f:
                        cookies = json.load(f)
                    for c in cookies:
                        session.cookies.set(c['name'], c['value'], domain=c.get('domain', ''))
                    print(f"✅ Loaded cookies from {json_path} (JSON)")
                    cookies_loaded = True
                except Exception as e:
                    print(f"⚠️ JSON format failed: {e}")
        
        if not cookies_loaded:
            print("⚠️ Could not load cookies from any format")
    
    return session


def get_product_info(product_url: str, session: requests.Session) -> Dict:
    """
    Get basic product information from Lazada.
    Returns dict with name, price, image, etc.
    """
    item_id = extract_item_id(product_url)
    if not item_id:
        return {"error": "Invalid URL - cannot extract item_id"}
    
    # Try to get product info from the page
    try:
        response = session.get(product_url, timeout=30)
        if response.status_code == 200:
            # Extract JSON data from page
            match = re.search(r'window\.__INIT_STATE__\s*=\s*({.*?});', response.text)
            if match:
                data = json.loads(match.group(1))
                # Extract product info from the data structure
                product_data = data.get('productData', {})
                return {
                    'item_id': item_id,
                    'name': product_data.get('title', f'Product {item_id}'),
                    'price': product_data.get('price', 'N/A'),
                    'image': product_data.get('image', ''),
                    'url': product_url
                }
    except Exception as e:
        print(f"⚠️ Error getting product info: {e}")
    
    # Fallback
    return {
        'item_id': item_id,
        'name': f'Lazada Product {item_id}',
        'price': 'N/A',
        'image': '',
        'url': product_url
    }


def crawl_reviews(
    product_url: str,
    cookies_path: Optional[str] = None,
    max_reviews: int = 100,
    delay_min: float = 2.0,
    delay_max: float = 5.0,
    item_id: Optional[str] = None  # NEW: allow passing item_id directly
) -> Tuple[List[Dict], str]:
    """
    Crawl reviews from Lazada product page.
    Crawls reviews from all rating levels (1-5 stars) to get diverse reviews.
    
    Args:
        product_url: Lazada product URL
        cookies_path: Path to cookies file (Netscape format)
        max_reviews: Maximum number of reviews to crawl
        delay_min: Minimum delay between requests (seconds)
        delay_max: Maximum delay between requests (seconds)
        item_id: Product item_id (optional, will extract from URL if not provided)
    
    Returns:
        Tuple of (list of reviews, error message if any)
    """
    # Use provided item_id or extract from URL
    if not item_id:
        item_id = extract_item_id(product_url)
    
    if not item_id:
        return [], "Không tìm thấy item_id trong URL!"
    
    print(f"🚀 Bắt đầu crawl reviews cho item_id: {item_id}")

    
    session = create_session(cookies_path)
    all_reviews = []
    
    # Lazada Review API endpoint
    api_url = "https://my.lazada.vn/pdp/review/getReviewList"
    
    # Filter values based on testing:
    # 1 = 1-star, 3 = 3-star, 4 = 4-star, 5 = 5-star
    # Filter 2 seems to also return 1-star reviews
    rating_filters = [1, 3, 4, 5]  # Focus on 1, 3, 4, 5 star reviews
    reviews_per_rating = max(max_reviews // 4, 15)  # Distribute across 4 ratings
    
    for rating_filter in rating_filters:
        if len(all_reviews) >= max_reviews:
            break
            
        print(f"⭐ Crawling {rating_filter}-star reviews...")
        page = 1
        rating_reviews = 0
        
        while rating_reviews < reviews_per_rating and len(all_reviews) < max_reviews:
            params = {
                "itemId": item_id,
                "pageSize": 50,
                "page": page,
                "filter": str(rating_filter),  # Filter by star rating
                "sort": "0"  # Default sort
            }
            
            try:
                response = session.get(api_url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"❌ HTTP {response.status_code} – Bỏ qua rating {6 - rating_filter}")
                    break
                
                data = response.json()
                items = data.get("model", {}).get("items", [])
                
                if not items:
                    break
                
                # Process each review
                for item in items:
                    review = {
                        'reviewContent': item.get('reviewContent', ''),
                        'rating': item.get('rating', 0),
                        'reviewTime': item.get('reviewTime', ''),
                        'buyerName': item.get('buyerName', ''),
                        'skuInfo': item.get('skuInfo', ''),
                        'images': item.get('images', []),
                        'likeCount': item.get('likeCount', 0)
                    }
                    all_reviews.append(review)
                    rating_reviews += 1
                    
                    if len(all_reviews) >= max_reviews:
                        break
                
                page += 1
                time.sleep(random.uniform(delay_min, delay_max))
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Lỗi request: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"❌ Lỗi parse JSON: {e}")
                break
            except Exception as e:
                print(f"❌ Lỗi không xác định: {e}")
                break
        
        print(f"  → Đã lấy {rating_reviews} reviews {6 - rating_filter} sao")
    
    if not all_reviews:
        return [], "Không lấy được review nào. Có thể cần cookies hợp lệ."
    
    # Shuffle to mix reviews from different ratings
    random.shuffle(all_reviews)
    
    print(f"✅ HOÀN TẤT! Đã crawl {len(all_reviews)} reviews (đa dạng rating)")
    return all_reviews[:max_reviews], ""


def reviews_to_dataframe(reviews: List[Dict]) -> pd.DataFrame:
    """
    Convert list of reviews to pandas DataFrame.
    """
    if not reviews:
        return pd.DataFrame()
    
    df = pd.DataFrame(reviews)
    
    # Convert timestamp if present
    if 'reviewTime' in df.columns:
        df['reviewTime'] = pd.to_datetime(df['reviewTime'], unit='ms', errors='coerce')
    
    return df


# Demo/test function
if __name__ == "__main__":
    test_url = "https://www.lazada.vn/products/pdp-i2633384520-s12855671277.html"
    print(f"Testing with URL: {test_url}")
    print(f"Item ID: {extract_item_id(test_url)}")
    
    # Try without cookies (may fail due to auth)
    reviews, error = crawl_reviews(test_url, max_reviews=10)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Got {len(reviews)} reviews")
        if reviews:
            print(f"Sample: {reviews[0]['reviewContent'][:100]}...")
