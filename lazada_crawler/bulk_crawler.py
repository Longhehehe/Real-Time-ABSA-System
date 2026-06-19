"""
Automated Bulk Crawler - Crawls reviews for 100 popular product keywords
This script runs independently and saves results to Excel files
"""

import time
import random
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import os

# 100 từ khóa tìm kiếm sản phẩm phổ biến
KEYWORDS = [
    # Thời trang & Phụ kiện (20)
    "áo thun nam", "áo sơ mi nữ", "quần jean", "váy đầm", "giày sneaker",
    "túi xách nữ", "đồng hồ nam", "kính mát", "nón lưỡi trai", "balo laptop",
    "dép sandal", "giày cao gót", "áo khoác", "quần short", "đầm maxi",
    "ví da nam", "thắt lưng", "khẩu trang", "vớ nam", "áo polo",
    
    # Điện tử & Công nghệ (20)
    "tai nghe bluetooth", "sạc dự phòng", "ốp lưng iphone", "cáp sạc type c", "loa bluetooth",
    "chuột gaming", "bàn phím cơ", "webcam", "USB 32GB", "thẻ nhớ 64GB",
    "đèn LED bàn học", "quạt mini", "máy hút bụi mini", "đồng hồ thông minh", "camera hành trình",
    "micro thu âm", "giá đỡ điện thoại", "kính VR", "đèn pin", "remote tivi",
    
    # Làm đẹp & Sức khỏe (20)
    "son môi", "kem chống nắng", "sữa rửa mặt", "serum vitamin c", "mặt nạ",
    "dầu gội", "sữa tắm", "kem dưỡng da", "nước hoa", "mascara",
    "phấn phủ", "kẻ mắt", "son dưỡng", "tẩy trang", "kem trị mụn",
    "máy cạo râu", "máy sấy tóc", "máy uốn tóc", "bàn chải điện", "vitamin tổng hợp",
    
    # Nhà cửa & Đời sống (20)
    "chảo chống dính", "nồi cơm điện", "bình giữ nhiệt", "hộp đựng thực phẩm", "dao nhà bếp",
    "thớt gỗ", "ly thủy tinh", "chén bát", "khăn tắm", "ga giường",
    "gối ngủ", "chăn mền", "rèm cửa", "đèn ngủ", "móc treo quần áo",
    "hộp đựng giày", "tủ vải", "ghế xếp", "thảm lau chân", "bình xịt nước",
    
    # Mẹ & Bé (10)
    "sữa bột", "tã bỉm", "bình sữa", "xe đẩy em bé", "đồ chơi trẻ em",
    "quần áo trẻ em", "bỉm dán", "núm ti giả", "ghế ăn dặm", "địu em bé",
    
    # Thể thao & Du lịch (10)
    "giày chạy bộ", "quần áo thể thao", "bóng đá", "vợt cầu lông", "găng tay gym",
    "thảm yoga", "dây nhảy", "bình nước thể thao", "vali kéo", "lều cắm trại"
]

class BulkCrawler:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("detach", True)
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)
        
        # Create output directory
        self.output_dir = "bulk_crawl_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def search_product(self, keyword):
        """Search for a product and return top product links"""
        print(f"\n🔍 Searching for: {keyword}")
        
        self.driver.get("https://www.lazada.vn")
        time.sleep(2)
        
        try:
            search_box = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search']")))
            search_box.clear()
            search_box.send_keys(keyword)
            search_box.send_keys(Keys.RETURN)
            time.sleep(3)
            
            # Scroll to load products
            for i in range(3):
                self.driver.execute_script(f"window.scrollTo(0, {(i+1)*800});")
                time.sleep(0.5)
            
            # Get product links
            product_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/products/']")
            
            links = []
            for link in product_links[:5]:  # Top 5 products per keyword
                href = link.get_attribute("href")
                if href and '/products/' in href and href not in links:
                    links.append(href)
            
            print(f"  Found {len(links)} product links")
            return links[:5]
            
        except Exception as e:
            print(f"  Error searching: {e}")
            return []
    
    def crawl_reviews(self, product_url, max_reviews=20):
        """Crawl reviews from a product page"""
        print(f"  📝 Crawling reviews from: {product_url[:60]}...")
        
        try:
            self.driver.get(product_url)
            time.sleep(3)
            
            # Scroll to reviews section
            for _ in range(10):
                self.driver.execute_script("window.scrollBy(0, 400);")
                time.sleep(0.3)
                try:
                    self.driver.find_element(By.ID, "module_product_review")
                    break
                except:
                    pass
            
            time.sleep(2)
            
            reviews = []
            
            # Find review items
            try:
                review_items = self.driver.find_elements(By.CSS_SELECTOR, ".mod-reviews .item, div[class*='review-item']")
                
                for item in review_items[:max_reviews]:
                    try:
                        content = ""
                        try:
                            content = item.find_element(By.CSS_SELECTOR, "div[class*='content']").text
                        except:
                            pass
                        
                        if not content:
                            continue
                        
                        author = "Anonymous"
                        try:
                            author = item.find_element(By.CSS_SELECTOR, "span[class*='name'], div[class*='middle'] span").text
                        except:
                            pass
                        
                        review_time = ""
                        try:
                            review_time = item.find_element(By.CSS_SELECTOR, "span[class*='date'], span[class*='time']").text
                        except:
                            pass
                        
                        reviews.append({
                            "reviewContent": content,
                            "buyerName": author,
                            "reviewTime": review_time,
                            "productUrl": product_url
                        })
                        
                    except Exception as e:
                        continue
                
            except Exception as e:
                print(f"    Error extracting reviews: {e}")
            
            print(f"    Collected {len(reviews)} reviews")
            return reviews
            
        except Exception as e:
            print(f"    Error crawling: {e}")
            return []
    
    def run(self, start_index=0, end_index=None):
        """Run the bulk crawl"""
        if end_index is None:
            end_index = len(KEYWORDS)
        
        keywords_to_crawl = KEYWORDS[start_index:end_index]
        
        print(f"\n{'='*60}")
        print(f"🚀 BULK CRAWLER - Starting crawl for {len(keywords_to_crawl)} keywords")
        print(f"{'='*60}\n")
        
        all_reviews = []
        
        for i, keyword in enumerate(keywords_to_crawl, start_index + 1):
            print(f"\n[{i}/{end_index}] Keyword: {keyword}")
            print("-" * 40)
            
            try:
                # Search for products
                product_links = self.search_product(keyword)
                
                # Crawl reviews from each product
                for link in product_links:
                    reviews = self.crawl_reviews(link)
                    for review in reviews:
                        review["keyword"] = keyword
                    all_reviews.extend(reviews)
                    
                    # Random delay between products
                    time.sleep(random.uniform(1, 3))
                
                # Save progress every 10 keywords
                if i % 10 == 0:
                    self.save_results(all_reviews, f"progress_{i}")
                    print(f"\n💾 Progress saved! Total reviews: {len(all_reviews)}")
                
                # Random delay between keywords
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                print(f"  ❌ Error with keyword '{keyword}': {e}")
                continue
        
        # Final save
        self.save_results(all_reviews, "final")
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETE! Total reviews collected: {len(all_reviews)}")
        print(f"{'='*60}")
        
        return all_reviews
    
    def save_results(self, reviews, suffix=""):
        """Save results to Excel"""
        if not reviews:
            return
        
        df = pd.DataFrame(reviews)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/lazada_reviews_{suffix}_{timestamp}.xlsx"
        df.to_excel(filename, index=False)
        print(f"  📁 Saved to: {filename}")
    
    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAZADA BULK CRAWLER - 100 Keywords")
    print("="*60)
    
    # You can customize the range here
    START_INDEX = 0  # Start from keyword index 0
    END_INDEX = 10   # End at keyword index 10 (first 10 keywords for testing)
    
    print(f"\nWill crawl keywords from index {START_INDEX} to {END_INDEX}")
    print(f"Keywords: {KEYWORDS[START_INDEX:END_INDEX]}")
    print("\nStarting in 5 seconds...")
    time.sleep(5)
    
    crawler = BulkCrawler()
    
    try:
        # Run the crawl
        reviews = crawler.run(START_INDEX, END_INDEX)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\nClosing browser...")
        # Don't close browser immediately - leave it open for user to see
        # crawler.close()
