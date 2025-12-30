#!/usr/bin/env python3
"""
👑 King AI v2 - Imperial Control Center
The unified command interface for deploying and managing the autonomous empire.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import webbrowser
import platform
from datetime import datetime
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.absolute()
CONFIG_FILE = ROOT_DIR / "scripts" / "control_config.json"
PEM_GLOB = "*.pem"
DEFAULT_IP = "ec2-13-222-9-32.compute-1.amazonaws.com"

# --- Visuals ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def header():
    clear_screen()
    print("""
\033[96m╔══════════════════════════════════════════════════════════════════╗
║               👑 KING AI v2 - IMPERIAL CONTROL                   ║
║             "The Empire Builds Itself While You Sleep"           ║
╚══════════════════════════════════════════════════════════════════╝\033[0m
""")

def log(msg, type="INFO"):
    colors = {
        "INFO": "\033[94m[ℹ️ INFO]\033[0m",
        "SUCCESS": "\033[92m[✅ SUCCESS]\033[0m",
        "WARN": "\033[93m[⚠️ WARN]\033[0m",
        "ERROR": "\033[91m[❌ ERROR]\033[0m",
        "ACTION": "\033[95m[🚀 ACTION]\033[0m"
    }
    print(f"{colors.get(type, type)} {msg}")

# --- Utilities ---
def save_config(ip):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"aws_ip": ip, "last_used": str(datetime.now())}, f)

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def find_key_file():
    keys = list(ROOT_DIR.glob(PEM_GLOB))
    if not keys:
        log("No .pem file found in project root!", "ERROR")
        sys.exit(1)
    return keys[0]

def run(cmd, cwd=None, capture=False):
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            cwd=cwd or ROOT_DIR,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True
        )
        return result.stdout.strip() if capture else True
    except subprocess.CalledProcessError as e:
        if capture:
            return None
        log(f"Command failed: {cmd}", "ERROR")
        sys.exit(1)

# --- Core Actions ---

def sync_secrets(ip, key_path):
    log("Syncing .env secrets to remote...", "ACTION")
    local_env = ROOT_DIR / ".env"
    if not local_env.exists():
        log("No .env file found via local path!", "WARN")
        return

    # Update dynamic variables in memory (or temp file)
    with open(local_env, 'r') as f:
        env_data = f.read()
    
    # For the remote server, we want to talk to Ollama locally (faster/reliable)
    ollama_line = "OLLAMA_URL=http://localhost:11434"
    if "OLLAMA_URL=" in env_data:
        import re
        env_data = re.sub(r'^OLLAMA_URL=.*$', ollama_line, env_data, flags=re.MULTILINE)
    else:
        env_data += f"\n{ollama_line}\n"

    temp_env = ROOT_DIR / ".env.deploy"
    with open(temp_env, 'w') as f:
        f.write(env_data)

    try:
        # Secure upload
        ssh_opts = f"-o StrictHostKeyChecking=no -i \"{key_path}\""
        run(f'ssh {ssh_opts} ubuntu@{ip} "mkdir -p ~/king-ai-v2"', capture=True)
        run(f'scp {ssh_opts} "{temp_env}" ubuntu@{ip}:~/king-ai-v2/.env', capture=True)
        log("Secrets deployed successfully.", "SUCCESS")
    finally:
        if temp_env.exists():
            os.remove(temp_env)

def deploy_code(ip, key_path):
    log("Packaging empire codebase...", "ACTION")
    
    # Clean old archives
    archive_name = "king-ai-deploy.tar.gz"
    if (ROOT_DIR / archive_name).exists():
        os.remove(ROOT_DIR / archive_name)

    # create tarball
    excludes = ["node_modules", "venv", "__pycache__", ".git", ".pytest_cache", ".env", "*.tar.gz"]
    exclude_args = " ".join([f'--exclude="{p}"' for p in excludes])
    
    # Reduced verbosity for tar
    run(f"tar -czf {archive_name} {exclude_args} .")
    
    log(f"Uploading {archive_name} to {ip}...", "ACTION")
    ssh_opts = f"-o StrictHostKeyChecking=no -i \"{key_path}\""
    run(f'scp {ssh_opts} "{archive_name}" ubuntu@{ip}:~/king-ai-v2.tar.gz')
    
    # Cleanup local tar
    os.remove(ROOT_DIR / archive_name)

    # Create robust deployment script
    deploy_script = """#!/bin/bash
set -e
echo "📂 Extracting codebase..."
mkdir -p king-ai-v2
tar -xzf king-ai-v2.tar.gz -C king-ai-v2
cd king-ai-v2

echo "🐍 Checking Python Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
fi

echo "📦 Installing Backend Dependencies..."
# Check for requirements.txt, otherwise install from pyproject.toml
if [ -f "src/requirements.txt" ]; then
    ./venv/bin/pip install -r src/requirements.txt
else
    ./venv/bin/pip install .
fi
./venv/bin/pip install psycopg2-binary

echo "💻 Installing Dashboard Dependencies..."
cd dashboard
npm install --silent
cd ..

echo "🗄️  Running Database Migrations..."
./venv/bin/alembic upgrade head

echo "🔄 Restarting Services..."
pkill -f uvicorn || true
pkill -f "npm run dev" || true

nohup ./venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
cd dashboard
nohup npm run dev -- --host 0.0.0.0 > frontend.log 2>&1 &

echo "✅ Services Launched!"
"""
    # Write script locally with Unix line endings and UTF-8 encoding
    script_path = ROOT_DIR / "deploy.sh"
    with open(script_path, "w", newline='\n', encoding='utf-8') as f:
        f.write(deploy_script)
        
    log("Uploading deployment instructions...", "ACTION")
    run(f'scp {ssh_opts} "{script_path}" ubuntu@{ip}:~/deploy.sh')
    os.remove(script_path)
    
    log("Executing remote deployment (Streaming Output)...", "ACTION")
    print("\033[90m" + "─" * 60 + "\033[0m")
    
    # Execute and Stream
    try:
        subprocess.run(
            f'ssh {ssh_opts} ubuntu@{ip} "chmod +x deploy.sh && ./deploy.sh"',
            shell=True,
            check=True
        )
        print("\033[90m" + "─" * 60 + "\033[0m")
        log("Deployment completed successfully.", "SUCCESS")
    except subprocess.CalledProcessError:
        print("\033[90m" + "─" * 60 + "\033[0m")
        log("Deployment failed. Check logs above.", "ERROR")

def main():
    header()
    
    # 1. Config & Setup
    config = load_config()
    saved_ip = config.get("aws_ip", DEFAULT_IP)
    
    print(f"\033[93mTarget Server:\033[0m {saved_ip}")
    new_ip = input(f"Press Enter to use current, or type new IP: ").strip()
    
    target_ip = new_ip if new_ip else saved_ip
    save_config(target_ip)
    
    key_file = find_key_file()
    log(f"Using Identity: {key_file.name}", "INFO")

    # 2. Main Menu
    print("\n\033[1mSelect Mission Profile:\033[0m")
    print(" [1] 🚀 Full Deployment (Code + Secrets + Restart)")
    print(" [2] 🔄 Quick Sync (Code Only)")
    print(" [3] 📺 View Remote Logs")
    print(" [4] 📡 Connect (SSH Shell)")
    print(" [q] Quit")
    
    choice = input("\nCommand > ").strip().lower()
    
    if choice == '1':
        sync_secrets(target_ip, key_file)
        deploy_code(target_ip, key_file)
    elif choice == '2':
        deploy_code(target_ip, key_file)
    elif choice == '3':
        header()
        log("Streaming backend logs (Ctrl+C to stop)...", "INFO")
        ssh_opts = f"-o StrictHostKeyChecking=no -i \"{key_file}\""
        try:
            run(f'ssh {ssh_opts} ubuntu@{target_ip} "tail -f king-ai-v2/backend.log"')
        except KeyboardInterrupt:
            pass
    elif choice == '4':
        ssh_opts = f"-o StrictHostKeyChecking=no -i \"{key_file}\""
        os.system(f'ssh {ssh_opts} ubuntu@{target_ip}')
        return

    # Post-Action Checks
    if choice in ['1', '2']:
        dashboard_url = f"http://{target_ip}:5173"
        log("Verifying Empire Status...", "INFO")
        time.sleep(2) # Give services a moment
        
        log(f"Empire is live at: {dashboard_url}", "SUCCESS")
        if input("Open dashboard? (Y/n): ").strip().lower() != 'n':
            webbrowser.open(dashboard_url)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\033[91mAborted by user.\033[0m")
