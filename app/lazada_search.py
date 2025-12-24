"""
Lazada Search - Search products on Lazada and return results
Works in Docker without GUI.
"""
import os
import re
import json
import requests
from typing import List, Dict, Optional
from urllib.parse import quote, urlencode


def search_lazada(
    keyword: str,
    page: int = 1,
    limit: int = 20,
    cookies_path: Optional[str] = None
) -> List[Dict]:
    """
    Search for products on Lazada.
    
    Args:
        keyword: Search keyword
        page: Page number (1-indexed)
        limit: Number of results to return
        cookies_path: Path to cookies file (optional)
    
    Returns:
        List of product dictionaries with name, price, url, image, rating
    """
    products = []
    
    try:
        # Create session with headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.lazada.vn/',
        })
        
        # Load cookies if available
        if cookies_path:
            _load_cookies(session, cookies_path)
        
        # Search API endpoint
        search_url = "https://www.lazada.vn/catalog/"
        
        params = {
            'q': keyword,
            'page': page,
            'ajax': 'true'
        }
        
        response = session.get(search_url, params=params, timeout=30)
        
        if response.status_code == 200:
            # Try to parse JSON from response
            try:
                data = response.json()
                items = data.get('mods', {}).get('listItems', [])
                
                for item in items[:limit]:
                    product = {
                        'name': item.get('name', 'Unknown'),
                        'price': item.get('priceShow', 'N/A'),
                        'original_price': item.get('originalPriceShow', ''),
                        'discount': item.get('discount', ''),
                        'url': 'https:' + item.get('productUrl', '') if item.get('productUrl', '').startswith('//') else item.get('productUrl', ''),
                        'image': item.get('image', ''),
                        'rating': item.get('ratingScore', 0),
                        'reviews': item.get('review', 0),
                        'sold': item.get('itemSoldCntShow', ''),
                        'item_id': item.get('itemId', '') or _extract_item_id(item.get('productUrl', '')),
                        'seller': item.get('sellerName', ''),
                        'location': item.get('location', '')
                    }
                    products.append(product)
                    
            except json.JSONDecodeError:
                # Fallback: parse HTML
                products = _parse_html_search(response.text, limit)
        
        else:
            print(f"Search failed with status {response.status_code}")
            
    except Exception as e:
        print(f"Search error: {e}")
    
    return products


def _load_cookies(session: requests.Session, cookies_path: str):
    """Load cookies into session."""
    try:
        # Try JSON format
        json_path = cookies_path.replace('.txt', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                cookies = json.load(f)
            for c in cookies:
                session.cookies.set(c['name'], c['value'], domain=c.get('domain', '.lazada.vn'))
            return
        
        # Try Netscape format
        if os.path.exists(cookies_path):
            import http.cookiejar as cookielib
            cj = cookielib.MozillaCookieJar(cookies_path)
            cj.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(cj)
    except Exception as e:
        print(f"Could not load cookies: {e}")


def _extract_item_id(url: str) -> str:
    """Extract item ID from Lazada URL."""
    if not url:
        return ''
    
    patterns = [
        r'-i(\d+)-s',           # Format: -i123456-s
        r'-i(\d+)\.',           # Format: -i123456.html
        r'-i(\d+)$',            # Format: ends with -i123456
        r'itemId=(\d+)',        # Query param: itemId=123456
        r'/i(\d+)\?',           # Format: /i123456?
        r'/i(\d+)$',            # Format: /i123456
        r'products/.*?-(\d+)\.html',  # Format: products/xxx-123456.html
        r'/(\d{6,})[-\.]',      # Any 6+ digit number followed by - or .
        r'/(\d{6,})\?',         # Any 6+ digit number followed by ?
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Last resort: find any 6+ digit number in URL
    all_numbers = re.findall(r'(\d{6,})', url)
    if all_numbers:
        return all_numbers[0]
    
    return ''


def _parse_html_search(html: str, limit: int) -> List[Dict]:
    """Fallback HTML parsing for search results."""
    products = []
    
    try:
        # Simple regex-based extraction
        # Look for product JSON in script tags
        pattern = r'"itemId":"(\d+)".*?"name":"([^"]+)".*?"priceShow":"([^"]+)".*?"productUrl":"([^"]+)"'
        matches = re.findall(pattern, html)
        
        for match in matches[:limit]:
            item_id, name, price, url = match
            products.append({
                'name': name,
                'price': price,
                'url': 'https:' + url if url.startswith('//') else url,
                'image': '',
                'rating': 0,
                'item_id': item_id
            })
    except:
        pass
    
    return products


def get_product_details(product_url: str, cookies_path: Optional[str] = None) -> Dict:
    """
    Get detailed info for a single product.
    
    Args:
        product_url: Lazada product URL
        cookies_path: Path to cookies file
    
    Returns:
        Product details dictionary
    """
    details = {
        'name': 'Unknown Product',
        'price': 'N/A',
        'image': '',
        'rating': 0,
        'reviews': 0,
        'description': ''
    }
    
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        
        if cookies_path:
            _load_cookies(session, cookies_path)
        
        response = session.get(product_url, timeout=30)
        
        if response.status_code == 200:
            html = response.text
            
            # Extract product name
            name_match = re.search(r'<title>([^|<]+)', html)
            if name_match:
                details['name'] = name_match.group(1).strip()
            
            # Extract price
            price_match = re.search(r'"priceShow":"([^"]+)"', html)
            if price_match:
                details['price'] = price_match.group(1)
            
            # Extract image
            img_match = re.search(r'"image":"([^"]+)"', html)
            if img_match:
                details['image'] = img_match.group(1)
            
            # Extract rating
            rating_match = re.search(r'"ratingScore":([\d.]+)', html)
            if rating_match:
                details['rating'] = float(rating_match.group(1))
            
    except Exception as e:
        print(f"Error getting product details: {e}")
    
    return details


# Test
if __name__ == "__main__":
    print("=== Lazada Search Test ===")
    
    results = search_lazada("dầu gội", limit=5)
    
    print(f"Found {len(results)} products:")
    for i, p in enumerate(results, 1):
        print(f"{i}. {p['name'][:50]}... - {p['price']}")
        print(f"   URL: {p['url'][:60]}...")
        print()
