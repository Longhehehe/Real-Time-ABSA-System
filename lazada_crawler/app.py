import streamlit as st
import pandas as pd
import time
import json
import os
from selenium.common.exceptions import NoSuchWindowException, WebDriverException
from crawler import LazadaCrawler
from utils import parse_cookie_string, save_to_csv

COOKIE_FILE = "lazada_cookies.json"

st.set_page_config(page_title="Lazada Crawler", layout="wide")

st.title("🛒 Lazada Review Crawler")

# Helper to load cookies from file
def load_cookies_from_file():
    if os.path.exists(COOKIE_FILE):
        try:
            with open(COOKIE_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

# Sidebar for Configuration
with st.sidebar:
    st.header("1. Configuration")
    
    # Debug mode toggle
    debug_mode = st.checkbox("🐛 Debug Mode (Save screenshots)", value=False, help="When enabled, saves screenshots at key points to help diagnose issues.")
    
    # Auto-load cookies on start
    if 'cookies' not in st.session_state:
        saved_cookies = load_cookies_from_file()
        if saved_cookies:
            st.session_state['cookies'] = saved_cookies
            st.success(f"Loaded {len(saved_cookies)} cookies from file.")
    
    st.markdown("### Option A: Manual Login (Recommended)")
    if st.button("Open Browser to Login"):
        # Helper to safely get or create crawler
        if 'crawler' not in st.session_state:
            st.session_state.crawler = LazadaCrawler(keep_alive=True, debug_mode=debug_mode)
        
        try:
            st.session_state.crawler.open_login_page()
            st.info("Browser opened! Please log in to Lazada manually in that window.")
        except (NoSuchWindowException, WebDriverException):
            st.warning("Previous browser window was closed. Re-opening new window...")
            st.session_state.crawler = LazadaCrawler(keep_alive=True, debug_mode=debug_mode)
            st.session_state.crawler.open_login_page()
            st.info("Browser opened!")
        
    if st.button("I have Logged In - Save Cookies"):
        if 'crawler' in st.session_state:
            try:
                cookies = st.session_state.crawler.get_cookies()
                st.session_state['cookies'] = cookies
                
                # Save to file
                with open(COOKIE_FILE, 'w') as f:
                    json.dump(cookies, f)
                    
                st.success(f"Captured and SAVED {len(cookies)} cookies to file! Next time they will be auto-loaded.")
            except Exception as e:
                st.error(f"Failed to get cookies: {e}")
        else:
            st.warning("Please open the browser first.")

    st.markdown("---")
    st.markdown("### Option B: Paste Cookies")
    cookie_input = st.text_area("Paste Cookie String", height=100, help="Alternative: Paste raw cookie string.")
    if st.button("Load Pasted Cookies"):
        if cookie_input:
            cookies = parse_cookie_string(cookie_input)
            st.session_state['cookies'] = cookies
            
            # Save to file
            with open(COOKIE_FILE, 'w') as f:
                json.dump(cookies, f)
                
            st.success(f"Parsed and SAVED {len(cookies)} cookies!")

# Main Area
tab1, tab2, tab3 = st.tabs(["Search & Crawl", "Data View", "Assisted Mode (Manual)"])

with tab2:
    st.header("📊 Data View")
    
    # Show current session data
    if 'crawled_data' in st.session_state and st.session_state['crawled_data'] is not None:
        df = st.session_state['crawled_data']
        st.success(f"✅ Có {len(df)} reviews trong session hiện tại")
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng reviews", len(df))
        with col2:
            if 'has_text_content' in df.columns:
                with_text = df['has_text_content'].sum() if df['has_text_content'].dtype == bool else len(df[df['has_text_content'] == True])
                st.metric("Có bình luận thực", int(with_text))
        with col3:
            if 'keyword' in df.columns:
                st.metric("Số từ khóa", df['keyword'].nunique())
        
        # Show dataframe
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="crawled_reviews.csv",
            mime="text/csv"
        )
    else:
        st.info("Chưa có dữ liệu crawl. Hãy chạy Bulk Crawl hoặc crawl sản phẩm trước!")
    
    # Show saved files
    st.write("---")
    st.subheader("📁 File đã lưu")
    
    import glob
    csv_files = glob.glob("*_reviews.csv") + glob.glob("bulk_crawl_*.csv")
    
    if csv_files:
        for f in csv_files[-10:]:  # Show last 10 files
            with st.expander(f"📄 {f}"):
                try:
                    df_file = pd.read_csv(f)
                    st.write(f"Rows: {len(df_file)}")
                    st.dataframe(df_file.head(5))
                except Exception as e:
                    st.error(f"Không đọc được file: {e}")
    else:
        st.info("Chưa có file CSV nào được lưu.")


with tab3:
    st.header("3. Assisted Mode (Semi-Automatic)")
    st.markdown("""
    **How to use:**
    1. Click "Open Browser" (in Sidebar).
    2. Manually navigate to the product page in the browser.
    3. Manually select the **Star Filter** (e.g., 5 Stars) and scroll to load comments.
    4. Come back here, select the Star Rating below, and click **"Scrape Visible Reviews"**.
    5. Go back to browser, click "Next Page", and repeat step 4.
    """)
    
    col_manual_1, col_manual_2 = st.columns(2)
    
    with col_manual_1:
         manual_star = st.selectbox("Assign Star Rating to captured data:", [5, 4, 3, 2, 1], index=0)
    
    if st.button("📸 Scrape Visible Reviews (No Clicks)"):
        if 'crawler' in st.session_state and st.session_state.crawler:
            try:
                # Call the new extraction method
                df_manual = st.session_state.crawler.extract_visible_reviews(override_star=manual_star)
                
                if not df_manual.empty:
                    st.success(f"Successfully scraped {len(df_manual)} reviews!")
                    
                    # Store in temporary session state
                    if 'manual_reviews' not in st.session_state:
                        st.session_state['manual_reviews'] = []
                    
                    # Add to temp list
                    st.session_state['manual_reviews'].extend(df_manual.to_dict('records'))
                    
                    # Show preview
                    st.dataframe(df_manual)
                else:
                    st.warning("No reviews found on current screen. Did you scroll down?")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please Open Browser first (Sidebar Option A).")

    # Display collected manual reviews
    if 'manual_reviews' in st.session_state and st.session_state['manual_reviews']:
        st.markdown("### Collected Data (Session)")
        df_collected = pd.DataFrame(st.session_state['manual_reviews'])
        st.dataframe(df_collected)
        st.text(f"Total collected: {len(df_collected)}")
        
        if st.button("💾 Save to Main Dataset"):
            if 'products' not in st.session_state:
                st.session_state.products = []
            
            # Create a 'Manual Crawl' product entry or append to existing?
            # Append rows to a CSV?
            # Let's save to a specific "Manual_Crawl.csv"
            save_to_csv(df_collected, "Manual_Crawl")
            
            # Also add to display session state
            st.success("Saved to `Manual_Crawl_reviews.csv`!")

with tab1:
    st.header("2. Search Product")
    keyword = st.text_input("Enter Product Name (e.g., iPhone 15)")
    
    if st.button("Search"):
        with st.spinner("Searching..."):
            try:
                # Use existing crawler if available, otherwise create new
                if 'crawler' in st.session_state:
                    crawler = st.session_state.crawler
                    # Check if alive by simple call
                    try:
                        _ = crawler.driver.current_url
                    except:
                        raise NoSuchWindowException("Driver died")
                        
                    # Update cookies if we have them in session state
                    if 'cookies' in st.session_state:
                        crawler.cookie_list = st.session_state['cookies']
                else:
                    crawler = LazadaCrawler(cookie_list=st.session_state.get('cookies'), keep_alive=True)
                    st.session_state.crawler = crawler

                results = crawler.search_product(keyword)
                st.session_state.products = results
                
                if not results:
                    st.error("No products found or search failed. Check debug screenshots in folder if needed.")
                else:
                    st.success(f"Found {len(results)} products.")
            
            except (NoSuchWindowException, WebDriverException):
                st.error("Browser was closed or disconnected. Please click Search again to restart.")
                st.session_state.pop('crawler', None) # Clear invalid crawler
                if 'products' in st.session_state:
                    del st.session_state.products
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

    # ============================================
    # BULK KEYWORDS AUTO-CRAWL SECTION
    # ============================================
    st.write("---")
    st.header("🔄 Bulk Keywords Auto-Crawl")
    st.markdown("Nhập danh sách từ khóa (mỗi dòng 1 từ khóa), app sẽ tự động crawl lần lượt.")
    
    # Default keywords - 80 popular Vietnamese products
    default_keywords = """áo thun nam
quần jean
giày sneaker
tai nghe bluetooth
sạc dự phòng
son môi
kem chống nắng
dầu gội
nồi cơm điện
bình giữ nhiệt
áo sơ mi nữ
váy đầm
túi xách nữ
đồng hồ nam
kính mát
nón lưỡi trai
balo laptop
dép sandal
giày cao gót
áo khoác
quần short
đầm maxi
ví da nam
thắt lưng
áo polo
cáp sạc type c
loa bluetooth
chuột gaming
bàn phím cơ
webcam
USB 32GB
thẻ nhớ 64GB
đèn LED bàn học
quạt mini
máy hút bụi mini
đồng hồ thông minh
camera hành trình
micro thu âm
giá đỡ điện thoại
sữa rửa mặt
serum vitamin c
mặt nạ
sữa tắm
kem dưỡng da
nước hoa
mascara
phấn phủ
kẻ mắt
tẩy trang
kem trị mụn
máy cạo râu
máy sấy tóc
máy uốn tóc
bàn chải điện
chảo chống dính
hộp đựng thực phẩm
dao nhà bếp
khăn tắm
ga giường
gối ngủ
chăn mền
đèn ngủ
sữa bột
tã bỉm
bình sữa
xe đẩy em bé
đồ chơi trẻ em
quần áo trẻ em
giày chạy bộ
quần áo thể thao
bóng đá
vợt cầu lông
găng tay gym
thảm yoga
dây nhảy
vali kéo
lều cắm trại
bàn học sinh
ghế văn phòng
máy tính bảng"""
    
    keywords_input = st.text_area("📝 Danh sách từ khóa (mỗi dòng 1 từ khóa)", 
                                   value=default_keywords, 
                                   height=200,
                                   help="Nhập nhiều từ khóa, mỗi từ khóa 1 dòng")
    
    col_bulk1, col_bulk2 = st.columns([1, 2])
    with col_bulk1:
        products_per_keyword = st.number_input("Số sản phẩm/từ khóa", min_value=1, max_value=20, value=3, 
                                                help="Số sản phẩm có nhiều reviews nhất sẽ được crawl")
    with col_bulk2:
        st.info("⚖️ **Balanced Mode**: Crawl 1,2,3 sao trước → sau đó 4,5 sao (số lượng bằng nhau)")
    
    if st.button("🚀 Bắt đầu Bulk Crawl", type="primary"):
        keywords_list = [kw.strip() for kw in keywords_input.strip().split('\n') if kw.strip()]
        
        if not keywords_list:
            st.error("Vui lòng nhập ít nhất 1 từ khóa!")
        else:
            st.info(f"Sẽ crawl {len(keywords_list)} từ khóa × {products_per_keyword} sản phẩm = {len(keywords_list) * products_per_keyword} sản phẩm (Balanced Mode)")
            
            # Initialize crawler
            if 'crawler' in st.session_state:
                crawler = st.session_state.crawler
                try:
                    _ = crawler.driver.current_url
                except:
                    crawler = LazadaCrawler(cookie_list=st.session_state.get('cookies'), keep_alive=True, debug_mode=debug_mode)
                    st.session_state.crawler = crawler
            else:
                crawler = LazadaCrawler(cookie_list=st.session_state.get('cookies'), keep_alive=True, debug_mode=debug_mode)
                st.session_state.crawler = crawler
            
            import random
            
            all_bulk_reviews = []
            total_keywords = len(keywords_list)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for kw_idx, keyword in enumerate(keywords_list):
                status_text.text(f"🔍 [{kw_idx+1}/{total_keywords}] Đang tìm kiếm: {keyword}")
                
                try:
                    # Search for this keyword
                    results = crawler.search_product(keyword)
                    
                    if not results:
                        st.warning(f"Không tìm thấy sản phẩm cho: {keyword}")
                        continue
                    
                    # Sort by reviews and take top N
                    sorted_results = sorted(results, key=lambda x: x.get('reviews', 0), reverse=True)
                    top_products = sorted_results[:products_per_keyword]
                    
                    st.text(f"  → Tìm thấy {len(results)} sản phẩm, crawl top {len(top_products)}")
                    
                    # Crawl each product
                    for prod_idx, product in enumerate(top_products):
                        status_text.text(f"📝 [{kw_idx+1}/{total_keywords}] {keyword} - Sản phẩm {prod_idx+1}/{len(top_products)}")
                        try:
                            df = crawler.crawl_reviews(product['link'], max_pages=-1, balanced_mode=True)
                            if df is not None and not df.empty:
                                df['keyword'] = keyword
                                df['product_title'] = product.get('title', '')
                                all_bulk_reviews.append(df)
                                st.text(f"    ✓ Thu được {len(df)} reviews")
                                
                                # === LƯU NGAY SAU MỖI SẢN PHẨM ===
                                # Update session_state để hiển thị ở Data View
                                current_df = pd.concat(all_bulk_reviews, ignore_index=True)
                                st.session_state['crawled_data'] = current_df
                                
                                # Lưu file CSV (ghi đè với dữ liệu mới nhất)
                                current_df.to_csv("bulk_crawl_progress.csv", index=False, encoding='utf-8-sig')
                                st.text(f"    💾 Đã lưu {len(current_df)} reviews → bulk_crawl_progress.csv")
                                
                        except Exception as e:
                            st.text(f"    ✗ Lỗi: {str(e)[:50]}")
                            # Longer delay on error to avoid being blocked
                            time.sleep(random.uniform(10, 20))
                        
                        # Random delay between products (tránh bị chặn)
                        delay = random.uniform(5, 12)
                        status_text.text(f"⏳ Đợi {delay:.1f}s trước sản phẩm tiếp theo...")
                        time.sleep(delay)
                    
                    # Random delay between keywords (dài hơn)
                    delay = random.uniform(8, 15)
                    status_text.text(f"⏳ Đợi {delay:.1f}s trước từ khóa tiếp theo...")
                    time.sleep(delay)
                    
                except Exception as e:
                    st.warning(f"Lỗi với từ khóa '{keyword}': {e}")
                    # Longer delay on keyword error
                    time.sleep(random.uniform(15, 30))
                
                progress_bar.progress((kw_idx + 1) / total_keywords)
            
            # Save all results
            if all_bulk_reviews:
                final_df = pd.concat(all_bulk_reviews, ignore_index=True)
                st.session_state['crawled_data'] = final_df
                
                # Save to file
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bulk_crawl_{timestamp}"
                save_to_csv(final_df, filename)
                
                st.success(f"✅ Hoàn thành! Thu được {len(final_df)} reviews từ {len(keywords_list)} từ khóa")
                st.balloons()
            else:
                st.warning("Không thu được reviews nào.")
    
    st.write("---")

    # Display Search Results
    if 'products' in st.session_state and st.session_state.products:
        st.write("---")
        
        # --- AUTO CRAWL SECTION ---
        st.header("🚀 Auto-Crawl")
        col_auto, col_config = st.columns([2, 2])
        
        with col_auto:
            auto_balanced_mode = st.checkbox("Balanced Mode (Auto-Crawl)", value=True, help="If checked, Auto-Crawl will limit high star reviews to match low star ones.")
            
            if st.button("Auto-Crawl Top 10 Products (Highest Reviews)"):
                st.info("Sorting products by review count...")
                # Sort products
                sorted_products = sorted(st.session_state.products, key=lambda x: x.get('reviews', 0), reverse=True)
                top_10 = sorted_products[:10]
                
                st.write("Top 10 Products identified:")
                for idx, p in enumerate(top_10):
                    st.text(f"{idx+1}. {p['title']} ({p.get('reviews', 0)} reviews)")
                    
                st.info("Starting Batch Crawl... This may take a while.")
                
                # Initialize crawler if needed
                if 'crawler' in st.session_state:
                    crawler = st.session_state.crawler
                else:
                    crawler = LazadaCrawler(cookie_list=st.session_state.get('cookies'), keep_alive=True)
                    st.session_state.crawler = crawler
                
                progress_bar = st.progress(0)
                all_crawled_reviews = []
                
                # Checkbox inside the column or use the one from manual section? No, distinct config.
                # Actually, we can't add a checkbox inside the button click event logic easily for *config* 
                # meant to be set BEFORE clicking. 
                # But I will look at where the button is.
                
                import random # Import locally if needed or ensure top level
                
                for i, product in enumerate(top_10):
                    st.text(f"Crawling {i+1}/10: {product['title']}...")
                    
                    # Random delay BEFORE crawling each product (except the first one maybe, or just always)
                    if i > 0:
                        wait_time = random.uniform(5, 10)
                        st.text(f"Waiting {wait_time:.1f}s before next product...")
                        time.sleep(wait_time)
                        
                    try:
                        # Use the auto_balanced_mode value
                        df = crawler.crawl_reviews(product['link'], max_pages=5, balanced_mode=auto_balanced_mode)
                        if not df.empty:
                            df['product_title'] = product['title']
                            all_crawled_reviews.append(df)
                    except Exception as e:
                        st.error(f"Failed to crawl {product['title']}: {e}")
                    
                    progress_bar.progress((i + 1) / 10)
                
                if all_crawled_reviews:
                    final_df = pd.concat(all_crawled_reviews, ignore_index=True)
                    st.session_state['crawled_data'] = final_df
                    st.success("Batch Crawling Completed!")
                else:
                    st.warning("No reviews crawled.")

        st.write("---")
        st.header("3. Select Product to Crawl (Manual)")
        
        for i, product in enumerate(st.session_state.products):
            price = product.get('price', 'N/A')
            reviews_count = product.get('reviews', 0)
            
            with st.expander(f"{product['title']} - {price} - ({reviews_count} reviews)"):
                st.write(f"Link: {product['link']}")
                
                # Input for pages to crawl
                col1, col2 = st.columns([2, 1])
                with col1:
                    max_pages = st.number_input(f"Max Pages (per star rating)", min_value=1, max_value=2000, value=5, key=f"pages_{i}")
                with col2:
                    crawl_all = st.checkbox("Crawl ALL pages", key=f"all_{i}")
                    balanced_mode = st.checkbox("Balanced (High=Low stars)", key=f"bal_{i}", help="Limits 4-5 stars to not exceed sum of 1-3 stars")

                if crawl_all:
                    final_pages = -1
                    st.caption("Will crawl until the end.")
                else:
                    final_pages = max_pages

                col_btn, col_append = st.columns([1, 2])
                with col_btn:
                    start_crawl_btn = st.button(f"Crawl Reviews", key=f"btn_{i}")
                with col_append:
                    append_mode = st.checkbox("Append to existing data", key=f"append_{i}", help="Keep current data and add new reviews to the bottom.")

                if start_crawl_btn:
                    st.info("Starting crawl...")
                    
                    progress_bar = st.progress(0)
                    # status_text = st.empty()
                    
                    try:
                        if 'crawler' in st.session_state:
                            crawler = st.session_state.crawler
                            try:
                                _ = crawler.driver.current_url
                            except:
                                raise NoSuchWindowException("Driver died")
                        else:
                            crawler = LazadaCrawler(cookie_list=st.session_state.get('cookies'), keep_alive=True)
                            st.session_state.crawler = crawler
                        
                        # Run crawling
                        # status_text.text("Navigating to product page...")
                        reviews_df = crawler.crawl_reviews(product['link'], max_pages=final_pages, balanced_mode=balanced_mode)
                        
                        if not reviews_df.empty:
                            if append_mode and 'crawled_data' in st.session_state and not st.session_state['crawled_data'].empty:
                                st.session_state['crawled_data'] = pd.concat([st.session_state['crawled_data'], reviews_df], ignore_index=True)
                                st.success(f"Successfully appended {len(reviews_df)} reviews! Total: {len(st.session_state['crawled_data'])}")
                            else:
                                st.session_state['crawled_data'] = reviews_df
                                st.success(f"Successfully crawled {len(reviews_df)} reviews!")
                                
                            if balanced_mode:
                                counts = reviews_df['star'].value_counts().sort_index()
                                st.write(counts)
                        else:
                            st.warning("No reviews found or crawling failed.")
                            
                    except NoSuchWindowException:
                        st.error("Browser window was closed. Please open browser again!")
                        if 'crawler' in st.session_state:
                           del st.session_state.crawler
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                    
                    progress_bar.progress(100)

with tab2:
    st.header("Crawled Data")
    if 'crawled_data' in st.session_state and not st.session_state['crawled_data'].empty:
        df = st.session_state['crawled_data']
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "lazada_reviews.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("No data crawled yet. Go to 'Search & Crawl' tab.")
