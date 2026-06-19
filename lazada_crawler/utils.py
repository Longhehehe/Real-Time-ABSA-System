import json

def parse_cookie_string(cookie_str):
    """
    Parses a cookie string (from browser header or Netscape format) into a list of dicts for Selenium.
    
    Args:
        cookie_str (str): Raw cookie string or JSON string.
        
    Returns:
        list: List of dicts [{'name': '...', 'value': '...'}, ...]
    """
    cookies = []
    
    # Try parsing as JSON first (if user pasted JSON)
    try:
        loaded = json.loads(cookie_str)
        if isinstance(loaded, list):
            return loaded
        elif isinstance(loaded, dict):
             # If it's a single dict, wrap it? Or maybe it's a key-value dict
             for k, v in loaded.items():
                 cookies.append({'name': k, 'value': str(v)})
             return cookies
    except json.JSONDecodeError:
        pass

    # Try parsing as HTTP Header string "key=value; key2=value2"
    if ';' in cookie_str or '=' in cookie_str:
        pairs = cookie_str.split(';')
        for pair in pairs:
            if '=' in pair:
                parts = pair.split('=', 1)
                name = parts[0].strip()
                value = parts[1].strip()
                cookies.append({'name': name, 'value': value})
    
    return cookies

def save_to_csv(data, filename):
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename
