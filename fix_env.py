import os

env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, "rb") as f:
        content = f.read()
    
    # Try different decodings
    decodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8-sig', 'utf-8', 'latin-1']
    text = None
    
    for enc in decodings:
        try:
            text = content.decode(enc)
            # Remove any leading BOM character if it survived
            if text.startswith('\ufeff'):
                text = text[1:]
            # If it's something like \xff\xfe at the start of the string
            if text.startswith('\xff\xfe') or text.startswith('ÿþ'):
                text = text[2:]
            
            # Check if it looks reasonable (not mostly nulls)
            if text.count('\x00') < len(text) / 10:
                print(f"Successfully decoded with {enc}")
                break
        except:
            continue
    
    if text:
        # Standardize line endings to LF for Linux compatibility
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        with open(env_path, "w", encoding='utf-8', newline='\n') as f:
            f.write(text)
        print("Standardized .env to UTF-8 (LF)")
    else:
        print("Failed to decode .env")
else:
    print(".env not found")
