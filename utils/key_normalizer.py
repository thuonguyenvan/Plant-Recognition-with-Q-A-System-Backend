"""
Vietnamese Key Normalizer
Converts snake_case English keys to proper Vietnamese
"""

KEY_MAPPING = {
    # Basic Plant Info
    "ten": "Tên",
    "ten_khac": "Tên khác", 
    "ten_khoa_hoc": "Tên khoa học",
    "ma_taxon": "Mã taxon",
    "ho": "Họ",
    
    # Section Names (already Vietnamese, keep as is)
    "Mô tả": "Mô tả",
    "Phân bố": "Phân bố",
    "Công dụng": "Công dụng",
    "Cách dùng": "Cách dùng",
    "Thành phần": "Thành phần",
    "Tính vị": "Tính vị",
    "Bộ phận dùng": "Bộ phận dùng",
    "Thông tin khác": "Thông tin khác",
    "luu_y": "Lưu ý",
    
    # Mô tả sub-keys
    "cây": "Mô tả cây",
    "lá": "Mô tả lá",
    "hoa": "Mô tả hoa",
    "quả": "Mô tả quả",
    "thân": "Mô tả thân",
    "rễ": "Mô tả rễ",
    "củ": "Mô tả củ",
    "mùa hoa": "Mùa hoa",
    "mùa quả": "Mùa quả",
    "cao": "Chiều cao",
    "mọc": "Cách mọc",
    
    # Phân bố sub-keys
    "phan_bo": "Phân bố địa lý",
    "phân_bố": "Phân bố địa lý",
    "thuong_moc": "Thường mọc",
    "thường_mọc": "Thường mọc",
    "Đặc điểm sinh trưởng": "Đặc điểm sinh trưởng",
    
    # Công dụng sub-keys
    "cong_dung_y_hoc": "Công dụng y học",
    "duoc_dung": "Được dùng",
    "được_dùng": "Được dùng",
    "làm thuốc": "Làm thuốc",
    "có tác dụng": "Có tác dụng",
    "chữa": "Chữa bệnh",
    
    # Cách dùng sub-keys
    "sắc uống": "Sắc uống",
    "giã nát": "Giã nát",
    "phơi khô": "Phơi khô",
    "liều dùng": "Liều dùng",
    
    # Thành phần sub-keys
    "thành phần": "Thành phần hóa học",
    "Các hợp chất": "Các hợp chất",
    "chứa": "Chứa chất",
    "đã nghiên cứu": "Đã nghiên cứu",
    "gồm chất": "Gồm chất",
    "tìm thấy": "Tìm thấy",
    "Ngoài ra": "Ngoài ra",
    
    # Bộ phận dùng
    "bo_phan_dung": "Bộ phận dùng",
    
    # Thông tin khác
    "thong_tin_khac": "Thông tin khác",
    
    # Tính vị
    "Có vị": "Có vị",
}


def normalize_key(key: str) -> str:
    """
    Normalize key to Vietnamese
    
    Args:
        key: Input key (English snake_case or Vietnamese)
        
    Returns:
        Normalized Vietnamese key
    """
    # Return mapped value if exists, otherwise return original
    return KEY_MAPPING.get(key, key)


def get_all_normalized_keys():
    """Get list of all possible normalized keys"""
    return list(set(KEY_MAPPING.values()))
