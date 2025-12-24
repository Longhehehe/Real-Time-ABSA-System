"""
Lazada Browser - Selenium browser for browsing and selecting products
Inject a floating button on product pages to add products to the list.
"""
import os
import sys
import json
import time
import re
from typing import Optional, Tuple, List
from threading import Thread

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# File to store selected products
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_PRODUCTS_FILE = os.path.join(BASE_DIR, 'selected_products.json')

# JavaScript to inject floating button
INJECT_BUTTON_JS = """
(function() {
    // Check if button already exists
    if (document.getElementById('streamlit-select-btn')) return;
    
    // Check if this is a product page
    var url = window.location.href;
    var isProductPage = url.includes('/products/') || url.includes('-i') && url.includes('-s');
    
    if (!isProductPage) return;
    
    // Create floating button container
    var container = document.createElement('div');
    container.id = 'streamlit-select-container';
    container.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        gap: 10px;
    `;
    
    // Create main select button
    var btn = document.createElement('button');
    btn.id = 'streamlit-select-btn';
    btn.innerHTML = '📌 Chọn sản phẩm này';
    btn.style.cssText = `
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        padding: 15px 25px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 50px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        transition: all 0.3s ease;
        font-family: 'Segoe UI', sans-serif;
    `;
    btn.onmouseover = function() {
        this.style.transform = 'scale(1.05)';
        this.style.boxShadow = '0 6px 20px rgba(255, 107, 107, 0.6)';
    };
    btn.onmouseout = function() {
        this.style.transform = 'scale(1)';
        this.style.boxShadow = '0 4px 15px rgba(255, 107, 107, 0.4)';
    };
    
    // Click handler - save product info
    btn.onclick = function() {
        var productInfo = extractProductInfo();
        if (productInfo) {
            saveProduct(productInfo);
            showNotification('✅ Đã thêm sản phẩm!');
            btn.innerHTML = '✅ Đã chọn!';
            btn.style.background = 'linear-gradient(135deg, #00C851, #007E33)';
            setTimeout(function() {
                btn.innerHTML = '📌 Chọn sản phẩm này';
                btn.style.background = 'linear-gradient(135deg, #FF6B6B, #FF8E53)';
            }, 2000);
        }
    };
    
    container.appendChild(btn);
    
    // Create status indicator
    var status = document.createElement('div');
    status.id = 'streamlit-status';
    status.style.cssText = `
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    `;
    status.innerHTML = '🔗 Kết nối với Streamlit';
    container.appendChild(status);
    
    document.body.appendChild(container);
    
    // Helper functions
    function extractProductInfo() {
        try {
            var name = document.querySelector('h1.pdp-mod-product-badge-title') || 
                       document.querySelector('[class*="product-title"]') ||
                       document.querySelector('h1');
            var price = document.querySelector('.pdp-price_type_normal') ||
                        document.querySelector('[class*="price"]');
            var image = document.querySelector('.gallery-preview-panel__content img') ||
                        document.querySelector('[class*="product"] img');
            
            return {
                url: window.location.href,
                name: name ? name.innerText.trim() : 'Unknown Product',
                price: price ? price.innerText.trim() : '',
                image: image ? image.src : '',
                timestamp: new Date().toISOString()
            };
        } catch(e) {
            return {
                url: window.location.href,
                name: document.title || 'Unknown Product',
                price: '',
                image: '',
                timestamp: new Date().toISOString()
            };
        }
    }
    
    function saveProduct(info) {
        // Store in localStorage for retrieval
        var products = JSON.parse(localStorage.getItem('streamlit_selected_products') || '[]');
        
        // Check if already added
        var exists = products.some(function(p) { return p.url === info.url; });
        if (!exists) {
            products.push(info);
            localStorage.setItem('streamlit_selected_products', JSON.stringify(products));
        }
        
        // Also trigger custom event for Selenium to catch
        window.dispatchEvent(new CustomEvent('productSelected', { detail: info }));
    }
    
    function showNotification(msg) {
        var notif = document.createElement('div');
        notif.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #00C851;
            color: white;
            padding: 15px 25px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            z-index: 9999999;
            animation: slideIn 0.3s ease;
            font-family: 'Segoe UI', sans-serif;
        `;
        notif.innerHTML = msg;
        document.body.appendChild(notif);
        setTimeout(function() { notif.remove(); }, 2000);
    }
})();
"""


def extract_item_id(url: str) -> Optional[str]:
    """Extract item ID from Lazada URL."""
    patterns = [
        r'-i(\d+)-s',
        r'itemId=(\d+)',
        r'/(\d+)\.html',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_selected_products() -> List[dict]:
    """Get list of products selected from browser."""
    if os.path.exists(SELECTED_PRODUCTS_FILE):
        try:
            with open(SELECTED_PRODUCTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_selected_products(products: List[dict]):
    """Save selected products to file."""
    with open(SELECTED_PRODUCTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(products, f, ensure_ascii=False, indent=2)


def clear_selected_products():
    """Clear all selected products."""
    if os.path.exists(SELECTED_PRODUCTS_FILE):
        os.remove(SELECTED_PRODUCTS_FILE)


class LazadaBrowser:
    """Selenium browser for Lazada with product selection."""
    
    def __init__(self):
        self.driver = None
        self.running = False
        self.selected_products = []
    
    def start(self, cookies_path: Optional[str] = None) -> bool:
        """Start the browser."""
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not installed!")
            return False
        
        try:
            print("🚀 Starting Lazada Browser...")
            
            # Setup Edge options
            options = EdgeOptions()
            options.add_argument("--start-maximized")
            options.add_argument("--disable-notifications")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            
            # Try to create driver - PRIORITY 1: Native Selenium Manager (Best for Edge)
            try:
                print("   Attempting to use native Selenium Manager...")
                self.driver = webdriver.Edge(options=options)
                print("✅ Native Selenium Manager worked!")
            except Exception as e1:
                print(f"⚠️ Native launch failed: {e1}")
                # PRIORITY 2: WebDriver Manager (Fallback)
                try:
                    print("   Attempting to use WebDriver Manager...")
                    service = EdgeService(EdgeChromiumDriverManager().install())
                    self.driver = webdriver.Edge(service=service, options=options)
                    print("✅ WebDriver Manager worked!")
                except Exception as e2:
                    print(f"❌ All methods failed. Error: {e2}")
                    return False
            
            # Load Lazada
            self.driver.get("https://www.lazada.vn")
            print("✅ Browser opened!")
            
            # Load cookies if provided
            if cookies_path and os.path.exists(cookies_path):
                self._load_cookies(cookies_path)
            
            self.running = True
            
            # Start monitoring thread
            monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            monitor_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting browser: {e}")
            return False
    
    def _load_cookies(self, cookies_path: str):
        """Load cookies from file."""
        try:
            # Try JSON format first
            json_path = cookies_path.replace('.txt', '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    cookies = json.load(f)
                for cookie in cookies:
                    try:
                        self.driver.add_cookie({
                            'name': cookie['name'],
                            'value': cookie['value'],
                            'domain': cookie.get('domain', '.lazada.vn')
                        })
                    except:
                        pass
                self.driver.refresh()
                print("✅ Cookies loaded!")
        except Exception as e:
            print(f"⚠️ Could not load cookies: {e}")
    
    def _monitor_loop(self):
        """Monitor browser and inject button on product pages."""
        last_url = ""
        
        while self.running and self.driver:
            try:
                current_url = self.driver.current_url
                
                # Inject button on new pages
                if current_url != last_url:
                    last_url = current_url
                    time.sleep(1)  # Wait for page load
                    self._inject_button()
                
                # Check for selected products
                self._check_selected_products()
                
                time.sleep(0.5)
                
            except Exception as e:
                if "disconnected" in str(e).lower() or "session" in str(e).lower():
                    self.running = False
                    break
                time.sleep(1)
    
    def _inject_button(self):
        """Inject the selection button into the page."""
        try:
            self.driver.execute_script(INJECT_BUTTON_JS)
        except:
            pass
    
    def _check_selected_products(self):
        """Check localStorage for newly selected products."""
        try:
            products_json = self.driver.execute_script(
                "return localStorage.getItem('streamlit_selected_products');"
            )
            if products_json:
                products = json.loads(products_json)
                if products != self.selected_products:
                    self.selected_products = products
                    save_selected_products(products)
                    print(f"📌 {len(products)} product(s) selected")
        except:
            pass
    
    def stop(self):
        """Stop the browser."""
        self.running = False
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
        print("🛑 Browser closed")
    
    def is_running(self) -> bool:
        """Check if browser is still running."""
        if not self.running or not self.driver:
            return False
        try:
            _ = self.driver.current_url
            return True
        except:
            self.running = False
            return False


# Global browser instance
_browser = None

def open_lazada_browser(cookies_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Open Lazada browser for product selection.
    
    Returns:
        Tuple of (success, message)
    """
    global _browser
    
    if _browser and _browser.is_running():
        return True, "Browser đã mở sẵn!"
    
    _browser = LazadaBrowser()
    if _browser.start(cookies_path):
        return True, "✅ Đã mở browser Lazada! Duyệt và click nút để chọn sản phẩm."
    else:
        return False, "❌ Không thể mở browser!"


def close_lazada_browser():
    """Close the Lazada browser."""
    global _browser
    if _browser:
        _browser.stop()
        _browser = None


def save_current_cookies(target_path: str = None) -> Tuple[bool, str]:
    """Save current browser cookies to file."""
    global _browser
    if not _browser or not _browser.driver:
        return False, "❌ Browser chưa mở! Hãy mở browser và đăng nhập trước."
    
    if target_path is None:
        # Default to project cookie path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_path = os.path.join(base_dir, 'app', 'cookie', 'lazada_cookies.txt')
        # Also try to save to root cookie/ if exists
        root_cookie = os.path.join(base_dir, 'cookie', 'lazada_cookies.txt')
        
    try:
        cookies = _browser.driver.get_cookies()
        if not cookies:
            return False, "⚠️ Không tìm thấy cookie nào! Đã đăng nhập chưa?"
            
        # Format as Netscape/JSON or just simplified list for requests?
        # The crawler uses `load_cookies` which supports Netscape format usually.
        # But let's check what verify_cookies expects.
        # Ideally, we save as Netscape format for compatibility with wget/curl/requests.
        
        # Simple Netscape format generator
        content = "# Netscape HTTP Cookie File\n"
        for c in cookies:
            domain = c.get('domain', '')
            flag = 'TRUE' if domain.startswith('.') else 'FALSE'
            path = c.get('path', '/')
            secure = 'TRUE' if c.get('secure') else 'FALSE'
            expiry = str(int(c.get('expiry', time.time() + 3600)))
            name = c.get('name', '')
            value = c.get('value', '')
            content += f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n"
            
        # Save to main path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Save to root path if needed
        if root_cookie and os.path.exists(os.path.dirname(root_cookie)):
             with open(root_cookie, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return True, f"✅ Đã lưu {len(cookies)} cookies! (Updated: {target_path})"
        
    except Exception as e:
        return False, f"❌ Lỗi khi lưu cookie: {str(e)}"


def is_browser_running() -> bool:
    """Check if browser is running."""
    global _browser
    return _browser is not None and _browser.is_running()


# Test
if __name__ == "__main__":
    print("=== Lazada Browser Test ===")
    success, msg = open_lazada_browser()
    print(msg)
    
    if success:
        print("Browser đang chạy. Nhấn Enter để lưu cookie...")
        input()
        s, m = save_current_cookies()
        print(m)
        print("Nhấn Enter để đóng...")
        input()
        close_lazada_browser()
