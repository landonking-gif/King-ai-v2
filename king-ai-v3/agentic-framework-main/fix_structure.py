#!/usr/bin/env python3
import os
import shutil

os.chdir('/home/ubuntu')

# Map of flat filenames to their proper paths
filemap = {}
for item in os.listdir('.'):
    if '\\' in item:
        parts = item.split('\\')
        if len(parts) >= 2:
            filemap[item] = '/'.join(parts)

print(f"Found {len(filemap)} files to reorganize")

# Create directory structure and move files
for flat_name, proper_path in sorted(filemap.items()):
    dirname = os.path.dirname(proper_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    try:
        shutil.move(flat_name, proper_path)
        print(f"✓ {flat_name[:50]}... → {proper_path[:60]}...")
    except Exception as e:
        print(f"✗ Failed: {flat_name[:40]}: {e}")

print("\nVerifying structure...")
os.system('ls -d agentic-framework-main/orchestrator agentic-framework-main/memory-service agentic-framework-main/mcp-gateway dashboard 2>&1')
