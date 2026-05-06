import urllib.request
import json

def fetch_library():
    url = "https://www.mp3quran.net/api/v3/reciters?language=eng"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print("Fetching from mp3quran.net...")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    
    lib = {}
    for r in data['reciters']:
        name = r['name'].replace(' ', '_').replace('/', '_').replace('-', '_')
        if r['moshaf']:
            # Pick the first available audio server
            server = r['moshaf'][0]['server']
            if not server.endswith('/'):
                server += '/'
            # Avoid naming collisions
            base_name = name
            counter = 1
            while base_name in lib.values() and server not in lib:
                base_name = f"{name}_{counter}"
                counter += 1
            lib[server] = base_name
            
    with open('reciters_lib.json', 'w', encoding='utf-8') as f:
        json.dump(lib, f, indent=4)
        
    print(f"Successfully generated reciters_lib.json with {len(lib)} Qaris!")

if __name__ == '__main__':
    fetch_library()
