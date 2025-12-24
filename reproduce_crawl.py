
import sys
import os
import requests
import json
import time

# Dynamic paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cookies_path = os.path.join(BASE_DIR, 'app', 'cookie', 'lazada_cookies.txt')
save_path = os.path.join(BASE_DIR, 'sort_analysis.json')
log_path = os.path.join(BASE_DIR, 'repro_sort_log.txt')

# Force UTF-8 for Windows Console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def load_cookies(session, path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    session.cookies.set(parts[5], parts[6], domain=parts[0], path=parts[2])
        return True
    except Exception as e:
        return False

def test_crawl():
    item_id = "2891823628" 
    api_url = "https://my.lazada.vn/pdp/review/getReviewList"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        'Referer': "https://www.lazada.vn/",
    })
    
    if not load_cookies(session, cookies_path):
        print("Failed to load cookies. Exiting.")
        return

    # Test Sort Orders 0-5
    sorts = [0, 1, 2, 3, 4, 5]
    results = {}
    
    for s in sorts:
        print(f"Testing Sort={s} ...")
        params = {
            "itemId": item_id,
            "pageSize": 20,
            "page": 1,
            "filter": "0",
            "sort": str(s)
        }
        
        try:
            resp = session.get(api_url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("model", {}).get("items", [])
                
                if items:
                    times = [item.get('reviewTime') for item in items]
                    ratings = [item.get('rating') for item in items]
                    
                    results[str(s)] = {
                        "count": len(items),
                        "first_rating": ratings[0],
                        "avg_rating": sum(ratings) / len(ratings),
                        "first_time": times[0],
                        "last_time": times[-1]
                    }
                else:
                    results[str(s)] = {"error": "No items"}
            else:
                results[str(s)] = {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            results[str(s)] = {"error": str(e)}
        
        time.sleep(1)
        
    with open("sort_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to sort_analysis.json")

if __name__ == "__main__":
    test_crawl()
