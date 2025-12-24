
import sys
import os
import time
import traceback

LOG_FILE = "browser_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

try:
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    log("🚀 Starting Debug Script")
    from selenium import webdriver
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.edge.options import Options as EdgeOptions
    log("✅ Selenium imported")
except ImportError as e:
    log(f"❌ Selenium import failed: {e}")
    sys.exit(1)

def test_edge():
    options = EdgeOptions()
    options.add_argument("--start-maximized")
    
    # Method 1: Selenium Manager (Built-in)
    log("\n--- Attempt 1: Native Selenium Manager ---")
    try:
        driver = webdriver.Edge(options=options)
        log("✅ Attempt 1 Success! Browser should be open.")
        driver.get("https://www.lazada.vn")
        log("   Navigated to Lazada")
        time.sleep(3)
        driver.quit()
        log("   Quit driver")
        return
    except Exception as e:
        log(f"❌ Attempt 1 Failed: {e}")
        log(traceback.format_exc())

    # Method 2: webdriver_manager
    log("\n--- Attempt 2: webdriver_manager ---")
    try:
        from webdriver_manager.microsoft import EdgeChromiumDriverManager
        path = EdgeChromiumDriverManager().install()
        log(f"📥 Driver downloaded to: {path}")
        service = EdgeService(path)
        driver = webdriver.Edge(service=service, options=options)
        log("✅ Attempt 2 Success!")
        driver.get("https://www.lazada.vn")
        time.sleep(3)
        driver.quit()
        return
    except Exception as e:
        log(f"❌ Attempt 2 Failed: {e}")
        log(traceback.format_exc())

    log("\n❌ ALL ATTEMPTS FAILED")

if __name__ == "__main__":
    test_edge()
