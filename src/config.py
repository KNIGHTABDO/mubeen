"""Mubeen configuration - Shazam for Quran Reciters"""

import os

# Paths
BASE_DIR = '/home/ubuntu/mubeen'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Audio settings
SAMPLE_RATE = 16000
CLIP_DURATION = 10  # seconds per clip
CLIP_OVERLAP = 0.5   # 50% overlap

# Model settings
ECAPA_MODEL = 'speechbrain/spkrec-ecapa-voxceleb'
EMBEDDING_DIM = 192

# FAISS settings
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, 'reciters.index')
FAISS_METADATA_PATH = os.path.join(EMBEDDINGS_DIR, 'reciters_meta.json')

# API settings
API_HOST = '0.0.0.0'
API_PORT = 8000
DASHBOARD_PORT = 8501

# Top Quran reciters to start with (name -> islamic.network code)
RECITERS = {
    'mishary_alafasy': 'ar.alafasy',
    'abdulrahman_sudais': 'ar.abdulrahmanalsudais',
    'saad_alghamdi': 'ar.saadalgamidi',
    'maher_almuaiqly': 'ar.maheralmuaiqly',
    'hani_arrifai': 'ar.hanirifai',
    'abdulbasit_abdulsamad': 'ar.abdulbasitabdulsamad',
    'ali_hudhaify': 'ar.alihuthaify',
    'mohammad_altablawi': 'ar.muhammadaltablawi',
    'yasser_aldossary': 'ar.yasseraldossari',
    'ahmed_ajmi': 'ar.ahmedajmi',
    'abobakr_alshtar': 'ar.abobakralshtar',
    'khalil_alhusary': 'ar.khalilalhusary',
    'saud_ashshuraim': 'ar.saudashshuraim',
    'fares_abbad': 'ar.faresabbad',
    'omar_alqazabri': 'ar.omaralqazabri',
    'nasser_alqatami': 'ar.nasseralqatami',
    'zaki_daghistani': 'ar.zakidaghistani',
    'abdullah_basfar': 'ar.abdullahbasfar',
    'muhammad_jibreel': 'ar.muhammadjibreel',
    'wadee_alyamani': 'ar.wadeealyamani',
}
