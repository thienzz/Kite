import os
import re

def replace_unicode(directory):
    replacements = {
        '[OK]': '[OK]',
        '[OK]': '[OK]',
        '[FILE]': '[FILE]',
        '[WARN]': '[WARN]',
        '[ERROR]': '[ERROR]',
        '[NEW]': '[NEW]',
        '[HOT]': '[HOT]',
        '[START]': '[START]',
        '[CHART]': '[CHART]',
        '[AI]': '[AI]',
        '[CHAT]': '[CHAT]',
        '[TOOL]': '[TOOL]',
        '[STORAGE]': '[STORAGE]',
        '[LINK]': '[LINK]',
        '[STOP]': '[STOP]',
        '[SAFE]': '[SAFE]',
    }
    
    # Also find all non-ascii and replace them if not in map
    non_ascii_re = re.compile(r'[^\x00-\x7F]')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    # First use the map for pretty names
                    for old, new in replacements.items():
                        new_content = new_content.replace(old, new)
                    
                    # Then catch any remaining non-ascii
                    new_content = non_ascii_re.sub(' ', new_content)
                    
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Sanitized: {path}")
                except Exception as e:
                    print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    import sys
    target = 'kite'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    replace_unicode(target)
