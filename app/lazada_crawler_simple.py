"""
Simple Lazada Review Crawler (Requests-based) with Balanced Mode
Works in Docker/Airflow without Selenium/Chrome.
Uses requests + Lazada API for fetching reviews.
"""
import os
import re
import json
import time
import random
import requests
from typing import List, Dict, Tuple, Optional


def crawl_reviews_simple(
    product_url: str,
    cookies_path: Optional[str] = None,
    max_reviews: int = 100,
    item_id: Optional[str] = None,
    balanced_mode: bool = False
) -> Tuple[List[Dict], Optional[str]]:
    """
    Crawl reviews from Lazada product page using requests.
    
    Args:
        product_url: Lazada product URL
        cookies_path: Path to cookies file (JSON format)
        max_reviews: Maximum reviews to fetch
        item_id: Product ID (will extract from URL if not provided)
        balanced_mode: If True, balance low-star (1-3) and high-star (4-5) reviews
    
    Returns:
        Tuple of (reviews list, error message if any)
    """
    reviews = []
    error = None
    
    # Extract item ID from URL if not provided
    if not item_id:
        patterns = [
            r'-i(\d+)-s', r'-i(\d+)\.', r'-i(\d+)$',
            r'itemId=(\d+)', r'/i(\d+)\?', r'/i(\d+)$',
            r'/(\d{6,})[-\.]'
        ]
        for pattern in patterns:
            match = re.search(pattern, product_url)
            if match:
                item_id = match.group(1)
                break
    
    if not item_id:
        return [], "Could not extract item ID from URL"
    
    print(f"📦 Crawling reviews for item: {item_id} (Balanced: {balanced_mode})")
    
    # Create session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.lazada.vn/',
    })
    
    # Load cookies if available
    if cookies_path:
        try:
            # Try JSON format first
            json_path = cookies_path.replace('.txt', '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    cookies = json.load(f)
                for c in cookies:
                    session.cookies.set(c.get('name', ''), c.get('value', ''), domain=c.get('domain', '.lazada.vn'))
                print(f"✅ Loaded cookies from {json_path}")
            elif os.path.exists(cookies_path):
                with open(cookies_path, 'r', encoding='utf-8') as f:
                    cookies = json.load(f)
                for c in cookies:
                    session.cookies.set(c.get('name', ''), c.get('value', ''), domain=c.get('domain', '.lazada.vn'))
                print(f"✅ Loaded cookies from {cookies_path}")
        except Exception as e:
            print(f"⚠️ Could not load cookies: {e}")
    
    if balanced_mode:
        reviews = _crawl_balanced(session, item_id, max_reviews)
    else:
        reviews, error = _crawl_all_reviews(session, item_id, max_reviews)
    
    print(f"✅ Total reviews crawled: {len(reviews)}")
    
    return reviews[:max_reviews], error


def _fetch_reviews_by_rating(session, item_id: str, rating: int, max_count: int, per_page: int = 50) -> List[Dict]:
    """Fetch reviews filtered by specific star rating."""
    reviews = []
    page = 1
    
    # Lazada API filter values: 1=1star, 2=2star, 3=3star, 4=4star, 5=5star
    # filter=0 means all reviews
    filter_value = rating if rating in [1, 2, 3, 4, 5] else 0
    
    while len(reviews) < max_count:
        try:
            api_url = "https://my.lazada.vn/pdp/review/getReviewList"
            params = {
                'itemId': item_id,
                'pageSize': per_page,
                'filter': filter_value,
                'sort': 0,
                'pageNo': page
            }
            
            response = session.get(api_url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"⚠️ API returned status {response.status_code} for {rating} star")
                break
            
            try:
                data = response.json()
            except:
                print(f"⚠️ Could not parse JSON for {rating} star")
                break
            
            model = data.get('model', {})
            review_items = model.get('items', [])
            
            if not review_items:
                break
            
            for item in review_items:
                review_text = item.get('reviewContent', '')
                if review_text and len(review_text.strip()) > 0:
                    reviews.append({
                        'review_text': review_text,
                        'reviewContent': review_text,  # For compatibility
                        'rating': item.get('rating', rating),
                        'review_time': item.get('reviewTime', ''),
                        'buyer_name': item.get('buyerName', 'Anonymous'),
                        'item_id': item_id,
                        'sku_info': item.get('skuInfo', ''),
                        'has_images': bool(item.get('images', [])),
                    })
            
            print(f"  📄 {rating}⭐ Page {page}: {len(review_items)} reviews, collected: {len(reviews)}/{max_count}")
            
            if len(reviews) >= max_count:
                break
            
            total_pages = model.get('pageCount', 1)
            if page >= total_pages:
                break
            
            page += 1
            time.sleep(random.uniform(0.5, 1.5))
            
        except Exception as e:
            print(f"❌ Error fetching {rating} star reviews: {e}")
            break
    
    return reviews[:max_count]


def _crawl_balanced(session, item_id: str, max_reviews: int) -> List[Dict]:
    """
    Crawl reviews with balanced mode:
    1. First crawl low-star reviews (1, 2, 3 stars)
    2. Then crawl high-star reviews (4, 5 stars) but limit to match low-star count
    """
    print("⚖️ Using BALANCED MODE: Prioritizing low-star reviews first")
    
    all_reviews = []
    low_star_reviews = []
    high_star_reviews = []
    
    # Calculate target per rating (divide by 5 ratings for initial estimate)
    per_rating_target = max(10, max_reviews // 5)
    
    # Step 1: Crawl LOW star reviews first (1, 2, 3 stars)
    print("\n📉 Phase 1: Crawling LOW star reviews (1-3 stars)...")
    for star in [1, 2, 3]:
        print(f"\n🔍 Fetching {star}⭐ reviews...")
        star_reviews = _fetch_reviews_by_rating(session, item_id, star, per_rating_target)
        low_star_reviews.extend(star_reviews)
        print(f"  ✓ Got {len(star_reviews)} reviews for {star}⭐")
        time.sleep(random.uniform(1, 2))
    
    low_star_count = len(low_star_reviews)
    print(f"\n📊 Total LOW star reviews collected: {low_star_count}")
    
    if low_star_count == 0:
        print("⚠️ No low-star reviews found. Crawling all reviews instead...")
        return _crawl_all_reviews(session, item_id, max_reviews)[0]
    
    # Step 2: Crawl HIGH star reviews (4, 5 stars) but LIMIT to match low-star count
    # We split low_star_count between 4 and 5 stars
    high_star_target = low_star_count  # Total high stars = total low stars
    per_high_star = max(5, high_star_target // 2)
    
    print(f"\n📈 Phase 2: Crawling HIGH star reviews (4-5 stars), target: {high_star_target} total...")
    for star in [4, 5]:
        remaining = high_star_target - len(high_star_reviews)
        if remaining <= 0:
            print(f"  ✓ Balanced reached! Skipping {star}⭐")
            break
            
        target = min(per_high_star, remaining)
        print(f"\n🔍 Fetching {star}⭐ reviews (target: {target})...")
        star_reviews = _fetch_reviews_by_rating(session, item_id, star, target)
        high_star_reviews.extend(star_reviews)
        print(f"  ✓ Got {len(star_reviews)} reviews for {star}⭐")
        time.sleep(random.uniform(1, 2))
    
    high_star_count = len(high_star_reviews)
    print(f"\n📊 Total HIGH star reviews collected: {high_star_count}")
    
    # Combine all reviews
    all_reviews = low_star_reviews + high_star_reviews
    
    print(f"\n✅ BALANCED RESULTS:")
    print(f"   Low stars (1-3): {low_star_count}")
    print(f"   High stars (4-5): {high_star_count}")
    print(f"   Total: {len(all_reviews)}")
    
    return all_reviews


def _crawl_all_reviews(session, item_id: str, max_reviews: int) -> Tuple[List[Dict], Optional[str]]:
    """Crawl all reviews without balancing."""
    reviews = []
    error = None
    page = 1
    per_page = 50
    
    while len(reviews) < max_reviews:
        try:
            api_url = "https://my.lazada.vn/pdp/review/getReviewList"
            params = {
                'itemId': item_id,
                'pageSize': per_page,
                'filter': 0,  # All reviews
                'sort': 0,
                'pageNo': page
            }
            
            response = session.get(api_url, params=params, timeout=30)
            
            if response.status_code != 200:
                break
            
            try:
                data = response.json()
            except:
                break
            
            model = data.get('model', {})
            review_items = model.get('items', [])
            
            if not review_items:
                break
            
            for item in review_items:
                review_text = item.get('reviewContent', '')
                if review_text and len(review_text.strip()) > 0:
                    reviews.append({
                        'review_text': review_text,
                        'reviewContent': review_text,
                        'rating': item.get('rating', 5),
                        'review_time': item.get('reviewTime', ''),
                        'buyer_name': item.get('buyerName', 'Anonymous'),
                        'item_id': item_id,
                        'sku_info': item.get('skuInfo', ''),
                        'has_images': bool(item.get('images', [])),
                    })
            
            print(f"📄 Page {page}: {len(review_items)} reviews, total: {len(reviews)}")
            
            if len(reviews) >= max_reviews:
                break
            
            total_pages = model.get('pageCount', 1)
            if page >= total_pages:
                break
            
            page += 1
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            error = f"Error: {e}"
            break
    
    return reviews[:max_reviews], error


# Legacy wrapper for backward compatibility
def crawl_reviews(
    product_url: str,
    cookies_path: Optional[str] = None,
    max_reviews: int = 100,
    delay_min: float = 1.0,
    delay_max: float = 2.0,
    item_id: Optional[str] = None,
    balanced_mode: bool = False,
    **kwargs
) -> Tuple[List[Dict], Optional[str]]:
    """Wrapper for backward compatibility"""
    return crawl_reviews_simple(product_url, cookies_path, max_reviews, item_id, balanced_mode)


# Test
if __name__ == "__main__":
    test_url = "https://www.lazada.vn/products/-i2581809925.html"
    print("Testing BALANCED mode:")
    reviews, err = crawl_reviews_simple(test_url, max_reviews=50, balanced_mode=True)
    print(f"\nGot {len(reviews)} reviews, error: {err}")
    
    # Count by rating
    from collections import Counter
    ratings = Counter(r['rating'] for r in reviews)
    print(f"Rating distribution: {dict(ratings)}")
