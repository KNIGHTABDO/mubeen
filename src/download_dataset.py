#!/usr/bin/env python3
"""Download full Quran surah-level audio from islamic.network CDN"""

import os
import sys
import time
import urllib.request
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from config import RAW_DIR
import json

# Target representative Surahs:
# 1 (Al-Fatiha - Short)
# 2 (Al-Baqarah - Extensively long)
# 19 (Maryam - Melodic variations)
# 36 (Ya-Sin - Intermediate flow)
# 112 (Al-Ikhlas - Short)
# 114 (An-Nas - Short)
TARGET_SURAHS = [1, 2, 18, 19, 20, 36, 55, 67, 78, 112, 114]
TOTAL_SURAHS = 114

def download_file(url, filepath):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 1000:
                return False
            with open(filepath, 'wb') as f:
                f.write(data)
            return True
    except Exception as e:
        return False

def download_reciter(code, name):
    reciter_dir = os.path.join(RAW_DIR, name)
    os.makedirs(reciter_dir, exist_ok=True)
    
    total = 0
    for surah in TARGET_SURAHS:
        filepath = os.path.join(reciter_dir, f'{surah:03d}.mp3')
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            total += 1
            continue
        
        # 'code' is now the server URL from mp3quran
        if code.startswith('http'):
            url = f"{code}{surah:03d}.mp3"
        else:
            url = f"https://cdn.islamic.network/quran/audio/128/{code}/{surah}.mp3"
            
        success = download_file(url, filepath)
        
        if success:
            total += 1
            if surah % 10 == 0:
                print(f'    {name}: {surah}/{TOTAL_SURAHS} surahs...')
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f'    {name}: FAILED surah {surah}')
        
        time.sleep(0.3)
    
    size_mb = sum(
        os.path.getsize(os.path.join(reciter_dir, f))
        for f in os.listdir(reciter_dir)
        if os.path.isfile(os.path.join(reciter_dir, f))
    ) / (1024 * 1024)
    
    print(f'  ✓ {name}: {total} surahs, {size_mb:.1f} MB')
    return total

def main():
    lib_path = '/home/ubuntu/mubeen/src/reciters_lib.json'
    if os.path.exists(lib_path):
        with open(lib_path) as f:
            reciters = json.load(f)
    else:
        reciters = {}
        
    print('=== Downloading Quran Audio Dataset ===')
    print(f'Reciters available in library: {len(reciters)}')
    print(f'Surahs per reciter target: {len(TARGET_SURAHS)}')
    print(f'Source: Mp3Quran / islamic.network CDN')
    print()
    
    grand_total = 0
    for code, name in reciters.items():
        count = download_reciter(code, name)
        grand_total += count
    
    print()
    print(f'=== Complete: {grand_total} total surahs ===')
    
    print()
    print('Dataset summary:')
    for name in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, name)
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.endswith('.mp3')])
            size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) / (1024*1024)
            print(f'  {name}: {count} surahs, {size:.1f} MB')
    
    total_size = sum(
        os.path.getsize(os.path.join(RAW_DIR, name, f))
        for name in os.listdir(RAW_DIR)
        for f in os.listdir(os.path.join(RAW_DIR, name))
        if os.path.isfile(os.path.join(RAW_DIR, name, f))
    ) / (1024*1024)
    print(f'\n  TOTAL: {total_size:.1f} MB')

if __name__ == '__main__':
    main()
