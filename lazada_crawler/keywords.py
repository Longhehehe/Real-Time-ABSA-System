# 100 từ khóa tìm kiếm sản phẩm phổ biến trên Lazada Vietnam

keywords = [
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

# Total: 100 keywords

if __name__ == "__main__":
    print(f"Total keywords: {len(keywords)}")
    for i, kw in enumerate(keywords, 1):
        print(f"{i}. {kw}")
