"""
Selenium Login Module
Automatically open browser for user to login and extract cookies.
"""
import os
import json
import time
from typing import Optional, Tuple
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.microsoft import EdgeChromiumDriverManager


# Cookie storage directory
COOKIE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cookie')

# Lazada URLs
LAZADA_LOGIN_URL = "https://member.lazada.vn/user/login"
LAZADA_HOME_URL = "https://www.lazada.vn/"


def setup_edge_driver(headless: bool = False) -> webdriver.Edge:
    """
    Setup Edge WebDriver with appropriate options.
    
    Args:
        headless: Run browser in headless mode (no UI)
    
    Returns:
        Edge WebDriver instance
    """
    options = Options()
    
    if headless:
        options.add_argument("--headless")
    
    # Common options to avoid detection
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1280,800")
    
    try:
        # Try using webdriver-manager first
        from webdriver_manager.microsoft import EdgeChromiumDriverManager
        service = Service(EdgeChromiumDriverManager().install())
        driver = webdriver.Edge(service=service, options=options)
    except Exception as e:
        print(f"⚠️ webdriver-manager failed: {e}")
        print("🔄 Trying default Edge driver...")
        # Fallback: let Selenium find EdgeDriver automatically
        driver = webdriver.Edge(options=options)
    
    return driver


def open_login_page(driver: webdriver.Chrome) -> bool:
    """
    Navigate to Lazada login page.
    
    Args:
        driver: Chrome WebDriver instance
    
    Returns:
        True if successful
    """
    try:
        driver.get(LAZADA_LOGIN_URL)
        time.sleep(2)  # Wait for page load
        return True
    except Exception as e:
        print(f"❌ Error opening login page: {e}")
        return False


def wait_for_login(driver: webdriver.Chrome, timeout: int = 300) -> bool:
    """
    Wait for user to complete login.
    Detects login by checking for specific elements or URL change.
    
    Args:
        driver: Chrome WebDriver instance
        timeout: Maximum wait time in seconds (default 5 minutes)
    
    Returns:
        True if login detected, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            current_url = driver.current_url
            
            # Check if redirected to home or member page
            if "lazada.vn" in current_url and "/user/login" not in current_url:
                # Additional check for logged-in state
                cookies = driver.get_cookies()
                # Look for session cookies
                session_cookies = [c for c in cookies if 'JSESSIONID' in c['name'] or 'lwid' in c['name']]
                if session_cookies:
                    print("✅ Login detected!")
                    return True
            
            time.sleep(2)
            
        except Exception as e:
            print(f"⚠️ Check error: {e}")
            time.sleep(2)
    
    return False


def extract_cookies(driver: webdriver.Chrome) -> list:
    """
    Extract all cookies from the browser.
    
    Args:
        driver: Chrome WebDriver instance
    
    Returns:
        List of cookie dictionaries
    """
    return driver.get_cookies()


def save_cookies_json(cookies: list, filepath: str) -> bool:
    """
    Save cookies to JSON file.
    
    Args:
        cookies: List of cookie dictionaries
        filepath: Path to save file
    
    Returns:
        True if successful
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, indent=2)
        print(f"✅ Cookies saved to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving cookies: {e}")
        return False


def save_cookies_netscape(cookies: list, filepath: str) -> bool:
    """
    Save cookies in Netscape format (compatible with requests).
    
    Args:
        cookies: List of cookie dictionaries
        filepath: Path to save file
    
    Returns:
        True if successful
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Netscape HTTP Cookie File\n")
            f.write("# This file was auto-generated by selenium_login.py\n\n")
            
            for cookie in cookies:
                domain = cookie.get('domain', '')
                # Netscape format: domain, flag, path, secure, expiry, name, value
                flag = "TRUE" if domain.startswith('.') else "FALSE"
                path = cookie.get('path', '/')
                secure = "TRUE" if cookie.get('secure', False) else "FALSE"
                expiry = str(int(cookie.get('expiry', 0)))
                name = cookie.get('name', '')
                value = cookie.get('value', '')
                
                f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
        
        print(f"✅ Cookies saved to {filepath} (Netscape format)")
        return True
    except Exception as e:
        print(f"❌ Error saving cookies: {e}")
        return False


def login_and_get_cookies(timeout: int = 300) -> Tuple[bool, str]:
    """
    Main function: Open browser, wait for login, save cookies.
    
    Args:
        timeout: Maximum wait time for login (seconds)
    
    Returns:
        Tuple of (success, cookie_filepath or error_message)
    """
    driver = None
    
    try:
        print("🚀 Khởi động trình duyệt Edge...")
        driver = setup_edge_driver(headless=False)
        
        print("📄 Mở trang đăng nhập Lazada...")
        if not open_login_page(driver):
            return False, "Không thể mở trang đăng nhập"
        
        print("⏳ Đợi bạn đăng nhập... (tối đa 5 phút)")
        print("   Sau khi đăng nhập xong, cookie sẽ tự động được lưu.")
        
        if not wait_for_login(driver, timeout):
            return False, "Timeout - không phát hiện đăng nhập"
        
        # Wait a bit more to ensure all cookies are set
        time.sleep(3)
        
        # Extract and save cookies
        cookies = extract_cookies(driver)
        
        if not cookies:
            return False, "Không lấy được cookies"
        
        # Save in both formats
        json_path = os.path.join(COOKIE_DIR, 'lazada_cookies.json')
        netscape_path = os.path.join(COOKIE_DIR, 'lazada_cookies.txt')
        
        save_cookies_json(cookies, json_path)
        save_cookies_netscape(cookies, netscape_path)
        
        print(f"✅ Đã lưu {len(cookies)} cookies!")
        return True, netscape_path
        
    except Exception as e:
        return False, f"Lỗi: {e}"
    
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def load_cookies_to_session(cookies_path: str, session) -> bool:
    """
    Load cookies from file and add to requests session.
    
    Args:
        cookies_path: Path to cookies file (JSON or Netscape)
        session: requests.Session instance
    
    Returns:
        True if successful
    """
    try:
        if cookies_path.endswith('.json'):
            with open(cookies_path, 'r', encoding='utf-8') as f:
                cookies = json.load(f)
            
            for cookie in cookies:
                session.cookies.set(
                    cookie['name'],
                    cookie['value'],
                    domain=cookie.get('domain', ''),
                    path=cookie.get('path', '/')
                )
        else:
            # Netscape format
            from http.cookiejar import MozillaCookieJar
            jar = MozillaCookieJar(cookies_path)
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies = jar
        
        return True
    except Exception as e:
        print(f"❌ Error loading cookies: {e}")
        return False


# Test function
if __name__ == "__main__":
    print("=== Lazada Cookie Extractor ===")
    success, result = login_and_get_cookies()
    
    if success:
        print(f"\n✅ Thành công! Cookies đã lưu tại: {result}")
    else:
        print(f"\n❌ Thất bại: {result}")
