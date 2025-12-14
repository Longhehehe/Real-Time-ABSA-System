"""
Trang Danh Sách Sản Phẩm - Quản lý sản phẩm để so sánh
"""
import streamlit as st
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import product_manager as pm
from lazada_crawler import extract_item_id, get_product_info, create_session

# Page Config
st.set_page_config(
    page_title="Danh Sách Sản Phẩm",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 { color: #00CC96; }
    .product-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #363945;
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🛒 Danh Sách Sản Phẩm So Sánh")
    
    # Initialize session state
    pm.init_session_state()
    
    # --- Sidebar: Cookies Configuration ---
    st.sidebar.header("🍪 Cấu hình Cookies")
    
    # Check if cookies already exist
    # In Docker: __file__ = /app/app/pages/1_xxx.py, project root = /app
    # We need to go up 3 levels: pages -> app -> project root
    cookies_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'cookie'
    )
    default_cookies = os.path.join(cookies_dir, 'lazada_cookies.txt')
    
    if os.path.exists(default_cookies):
        pm.set_cookies_path(default_cookies)
        st.sidebar.success("✅ Đã có cookies Lazada!")
        st.sidebar.caption(f"📁 {default_cookies}")
    
    # Option 1: Auto login with Selenium
    st.sidebar.markdown("### 🤖 Đăng nhập tự động")
    st.sidebar.markdown("Mở browser để đăng nhập Lazada, cookies sẽ tự động được lưu.")
    
    if st.sidebar.button("🔐 Mở Browser Đăng Nhập", use_container_width=True):
        st.sidebar.info("⏳ Đang mở trình duyệt...")
        
        try:
            from selenium_login import login_and_get_cookies
            
            with st.spinner("🚀 Đang mở trình duyệt Chrome..."):
                success, result = login_and_get_cookies(timeout=300)
            
            if success:
                pm.set_cookies_path(result)
                st.sidebar.success("✅ Đăng nhập thành công!")
                st.rerun()
            else:
                st.sidebar.error(f"❌ {result}")
        except ImportError as e:
            st.sidebar.error(f"❌ Thiếu thư viện Selenium. Chạy: pip install selenium webdriver-manager")
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi: {e}")
    
    st.sidebar.markdown("---")
    
    # Option 2: Manual upload
    st.sidebar.markdown("### 📤 Hoặc upload cookies thủ công")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload file cookies (.txt)",
        type=['txt'],
        help="File cookies Netscape format"
    )
    
    if uploaded_file:
        os.makedirs(cookies_dir, exist_ok=True)
        
        cookies_path = os.path.join(cookies_dir, 'lazada_cookies.txt')
        with open(cookies_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        pm.set_cookies_path(cookies_path)
        st.sidebar.success("✅ Đã upload cookies!")
        st.rerun()
    
    # Show current status
    st.sidebar.markdown("---")
    if pm.is_cookies_uploaded():
        st.sidebar.info(f"📁 Cookies: OK")
    else:
        st.sidebar.warning("⚠️ Chưa có cookies - có thể không crawl được")
    
    # --- Main Content ---
    st.markdown("---")
    
    # ===== Lazada Search Section =====
    st.subheader("🔍 Tìm Kiếm Sản Phẩm Lazada")
    
    # Search input
    search_col1, search_col2 = st.columns([4, 1])
    
    with search_col1:
        search_keyword = st.text_input(
            "Nhập từ khóa tìm kiếm",
            placeholder="Ví dụ: dầu gội, điện thoại, laptop...",
            key="search_keyword"
        )
    
    with search_col2:
        st.write("")  # Spacing
        search_button = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
    
    # Search and display results
    if search_button and search_keyword:
        try:
            from lazada_search import search_lazada
            
            with st.spinner(f"🔍 Đang tìm kiếm '{search_keyword}'..."):
                results = search_lazada(search_keyword, limit=12, cookies_path=pm.get_cookies_path())
            
            if results:
                st.success(f"✅ Tìm thấy {len(results)} sản phẩm")
                
                # Store results in session state
                st.session_state['search_results'] = results
            else:
                st.warning("❌ Không tìm thấy sản phẩm nào. Thử từ khóa khác.")
                
        except ImportError:
            st.error("❌ Module lazada_search chưa được cài đặt")
        except Exception as e:
            st.error(f"❌ Lỗi tìm kiếm: {e}")
    
    # Display search results
    if 'search_results' in st.session_state and st.session_state['search_results']:
        st.markdown("### 📋 Kết quả tìm kiếm")
        st.caption("Click **Thêm** để thêm sản phẩm vào danh sách so sánh")
        
        results = st.session_state['search_results']
        
        # Display in grid
        cols = st.columns(3)
        for idx, product in enumerate(results):
            with cols[idx % 3]:
                with st.container():
                    # Product card
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        border: 1px solid #0f3460;
                    ">
                        <h4 style="color: #e94560; margin: 0; font-size: 14px;">
                            {product.get('name', 'Unknown')[:40]}...
                        </h4>
                        <p style="color: #00ff88; font-size: 18px; margin: 5px 0; font-weight: bold;">
                            {product.get('price', 'N/A')}
                        </p>
                        <p style="color: #888; font-size: 12px; margin: 0;">
                            ⭐ {product.get('rating', 0)} | 🛒 {product.get('sold', '0')} đã bán
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add button
                    item_id = product.get('item_id', '')
                    if item_id and item_id not in pm.get_products():
                        if st.button("➕ Thêm", key=f"add_search_{idx}", use_container_width=True):
                            pm.add_product(
                                item_id=item_id,
                                name=product.get('name', f'Sản phẩm {item_id}'),
                                url=product.get('url', ''),
                                image=product.get('image', ''),
                                price=product.get('price', 'N/A')
                            )
                            st.success("✅ Đã thêm!")
                            st.rerun()
                    elif item_id in pm.get_products():
                        st.info("✓ Đã có trong danh sách")
        
        # Clear results button
        if st.button("🗑️ Xóa kết quả tìm kiếm"):
            st.session_state['search_results'] = []
            st.rerun()
    
    st.markdown("---")
    
    # ===== Manual Add Product Section =====
    st.subheader("➕ Thêm Sản Phẩm Bằng URL")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_url = st.text_input(
            "Nhập URL sản phẩm Lazada",
            placeholder="https://www.lazada.vn/products/pdp-i1216257-s1509400.html"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        add_button = st.button("➕ Thêm", type="primary", use_container_width=True)
    
    if add_button and product_url:
        item_id = extract_item_id(product_url)
        
        if not item_id:
            st.error("❌ URL không hợp lệ! Vui lòng nhập URL sản phẩm Lazada.")
        elif item_id in pm.get_products():
            st.warning("⚠️ Sản phẩm này đã có trong danh sách!")
        else:
            with st.spinner("Đang lấy thông tin sản phẩm..."):
                # Get product info
                session = create_session(pm.get_cookies_path())
                info = get_product_info(product_url, session)
                
                # Add to list
                pm.add_product(
                    item_id=item_id,
                    name=info.get('name', f'Sản phẩm {item_id}'),
                    url=product_url,
                    image=info.get('image', ''),
                    price=info.get('price', 'N/A')
                )
                
                st.success(f"✅ Đã thêm sản phẩm: {info.get('name', item_id)}")
                st.rerun()
    
    st.markdown("---")
    
    # Product List
    st.subheader("📋 Danh Sách Sản Phẩm")
    
    products = pm.get_products()
    
    if not products:
        st.info("📭 Chưa có sản phẩm nào. Hãy thêm ít nhất 2 sản phẩm để so sánh!")
    else:
        # Display products
        for item_id, product in products.items():
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    if product.image:
                        st.image(product.image, width=100)
                    else:
                        st.markdown("🖼️ No Image")
                
                with col2:
                    st.markdown(f"**{product.name}**")
                    st.markdown(f"💰 Giá: {product.price}")
                    st.markdown(f"🔗 [Xem trên Lazada]({product.url})")
                    st.caption(f"ID: {item_id}")
                
                with col3:
                    if st.button("🗑️ Xóa", key=f"del_{item_id}"):
                        pm.remove_product(item_id)
                        st.rerun()
                
                st.markdown("---")
    
    # Compare Button
    st.subheader("🎯 So Sánh Sản Phẩm")
    
    product_count = pm.get_product_count()
    
    if product_count < 2:
        st.warning(f"⚠️ Cần ít nhất 2 sản phẩm để so sánh (hiện có: {product_count})")
        st.button("📊 So Sánh", disabled=True, use_container_width=True)
    else:
        st.success(f"✅ Có {product_count} sản phẩm - Sẵn sàng so sánh!")
        
        if st.button("📊 Bắt Đầu So Sánh", type="primary", use_container_width=True):
            st.switch_page("pages/2_📊_So_Sánh.py")
    
    # Clear all button
    if products:
        st.markdown("---")
        if st.button("🗑️ Xóa Tất Cả", type="secondary"):
            pm.clear_products()
            st.rerun()


if __name__ == "__main__":
    main()
