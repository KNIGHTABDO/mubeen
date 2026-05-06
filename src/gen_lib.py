import json
import os

try:
    with open('/home/ubuntu/mubeen/all_editions.json') as f:
        data = json.load(f)
    
    reciters = {}
    for e in data['data']:
        if e['language'] == 'ar' and e['format'] == 'audio':
            # Clean name for filesystem
            name = e['englishName'].strip()
            name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('.', '')
            reciters[e['identifier']] = name
            
    with open('/home/ubuntu/mubeen/src/reciters_lib.json', 'w') as f:
        json.dump(reciters, f, indent=2)
    print(f'SUCCESS: Saved {len(reciters)} reciters')
except Exception as e:
    print(f'ERROR: {e}')
