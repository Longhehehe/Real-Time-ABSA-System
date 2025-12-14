"""
Product Manager Module
Manage comparison list of products in Streamlit session state.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import streamlit as st


@dataclass
class Product:
    """Product data class."""
    item_id: str
    name: str
    url: str
    image: str = ""
    price: str = "N/A"
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reviews: List[Dict] = field(default_factory=list)
    predictions: List[Dict] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


def init_session_state():
    """Initialize session state for product management."""
    if 'comparison_products' not in st.session_state:
        st.session_state['comparison_products'] = {}
    
    if 'cookies_uploaded' not in st.session_state:
        st.session_state['cookies_uploaded'] = False
    
    if 'cookies_path' not in st.session_state:
        st.session_state['cookies_path'] = None


def add_product(
    item_id: str,
    name: str,
    url: str,
    image: str = "",
    price: str = "N/A"
) -> bool:
    """
    Add a product to the comparison list.
    
    Args:
        item_id: Unique product identifier
        name: Product name
        url: Product URL
        image: Product image URL
        price: Product price
    
    Returns:
        True if added, False if already exists
    """
    init_session_state()
    
    if item_id in st.session_state['comparison_products']:
        return False  # Already exists
    
    product = Product(
        item_id=item_id,
        name=name,
        url=url,
        image=image,
        price=price
    )
    
    st.session_state['comparison_products'][item_id] = product
    return True


def remove_product(item_id: str) -> bool:
    """
    Remove a product from the comparison list.
    
    Args:
        item_id: Product identifier to remove
    
    Returns:
        True if removed, False if not found
    """
    init_session_state()
    
    if item_id in st.session_state['comparison_products']:
        del st.session_state['comparison_products'][item_id]
        return True
    return False


def get_products() -> Dict[str, Product]:
    """
    Get all products in the comparison list.
    
    Returns:
        Dict mapping item_id to Product
    """
    init_session_state()
    return st.session_state['comparison_products']


def get_product(item_id: str) -> Optional[Product]:
    """
    Get a specific product by ID.
    
    Args:
        item_id: Product identifier
    
    Returns:
        Product if found, None otherwise
    """
    init_session_state()
    return st.session_state['comparison_products'].get(item_id)


def get_product_count() -> int:
    """Get number of products in comparison list."""
    init_session_state()
    return len(st.session_state['comparison_products'])


def can_compare() -> bool:
    """Check if comparison is possible (>= 2 products)."""
    return get_product_count() >= 2


def clear_products():
    """Clear all products from comparison list."""
    init_session_state()
    st.session_state['comparison_products'] = {}


def update_product_reviews(item_id: str, reviews: List[Dict]) -> bool:
    """
    Update reviews for a product.
    
    Args:
        item_id: Product identifier
        reviews: List of review dicts
    
    Returns:
        True if updated, False if product not found
    """
    product = get_product(item_id)
    if product:
        product.reviews = reviews
        return True
    return False


def update_product_predictions(item_id: str, predictions: List[Dict]) -> bool:
    """
    Update predictions for a product.
    
    Args:
        item_id: Product identifier
        predictions: List of prediction dicts
    
    Returns:
        True if updated, False if product not found
    """
    product = get_product(item_id)
    if product:
        product.predictions = predictions
        return True
    return False


def update_product_scores(item_id: str, scores: Dict[str, float]) -> bool:
    """
    Update aggregated scores for a product.
    
    Args:
        item_id: Product identifier
        scores: Dict mapping aspect to score (0-100)
    
    Returns:
        True if updated, False if product not found
    """
    product = get_product(item_id)
    if product:
        product.scores = scores
        return True
    return False


def set_cookies_path(path: str):
    """Set the path to uploaded cookies file."""
    init_session_state()
    st.session_state['cookies_path'] = path
    st.session_state['cookies_uploaded'] = True


def get_cookies_path() -> Optional[str]:
    """Get the path to cookies file."""
    init_session_state()
    return st.session_state['cookies_path']


def is_cookies_uploaded() -> bool:
    """Check if cookies have been uploaded."""
    init_session_state()
    return st.session_state['cookies_uploaded']


def get_products_for_comparison() -> Dict[str, Dict[str, float]]:
    """
    Get products formatted for radar chart comparison.
    
    Returns:
        Dict mapping product name to scores dict
    """
    products = get_products()
    result = {}
    
    for item_id, product in products.items():
        if product.scores:
            result[product.name] = product.scores
    
    return result
