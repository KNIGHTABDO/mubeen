import streamlit as st
import os
import json
import time
import glob
import subprocess
import psutil
import pandas as pd
import plotly.express as px
from collections import Counter

# Page Config
st.set_page_config(page_title='Mubeen AI Control', page_icon='🕌', layout='wide')

# Custom CSS
st.markdown("""
    <style>
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 12px; border-left: 5px solid #4CAF50; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Helper: Safe JSON Load
def load_json(path):
    if not os.path.exists(path): return {}
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return {}

# Paths
BASE_DIR = "/home/ubuntu/mubeen"
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
META_PATH = os.path.join(BASE_DIR, "data/embeddings/reciters_meta.json")
STATUS_FILE = os.path.join(BASE_DIR, "src/embedding_status.json")
LIB_PATH = os.path.join(BASE_DIR, "src/reciters_lib.json")

# Load dynamic library
RECITERS_MAP = load_json(LIB_PATH)
if not RECITERS_MAP:
    RECITERS_MAP = {'ar.alafasy': 'Mishary_Alafasy'} # Fallback

# Sidebar
st.sidebar.title("🕌 Mubeen AI")
page = st.sidebar.selectbox("Navigate", ["Dashboard", "Train & Data Manager", "Identify (Test)", "System Settings"])

st.title("🕌 Mubeen Project Control Center")
st.divider()

if page == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    reciters_list = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))] if os.path.exists(RAW_DIR) else []
    meta = load_json(META_PATH)
    if not isinstance(meta, dict): meta = {}
    c1.metric("Reciters Indexed", len(meta.get("reciters", [])))
    c2.metric("Index Size", f"{meta.get('total_vectors', 0)} Vectors")
    is_running = subprocess.run(['pgrep', '-f', 'pipeline.py'], capture_output=True).stdout.strip()
    c3.metric("Task Status", "🟢 Running" if is_running else "⚪ Idle")
    disk = psutil.disk_usage('/')
    c4.metric("Storage Free", f"{disk.free / (1024**3):.1f} GB")

    st.subheader("📊 Dataset Distribution")
    if os.path.exists(RAW_DIR):
        reciter_data = []
        for name in sorted(os.listdir(RAW_DIR)):
            path = os.path.join(RAW_DIR, name)
            if os.path.isdir(path):
                files = len([f for f in os.listdir(path) if f.endswith('.mp3')])
                clips = len(glob.glob(os.path.join(PROCESSED_DIR, name, "*.wav"))) if os.path.exists(PROCESSED_DIR) else 0
                reciter_data.append({"Reciter": name, "Source MP3s": files, "Training Clips": clips})
        if reciter_data:
            df = pd.DataFrame(reciter_data)
            fig = px.bar(df, x="Reciter", y="Training Clips", color="Training Clips", template="plotly_dark", title="Total Training Samples")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Train & Data Manager":
    st.subheader("🧬 Master Pipeline: Download, Preprocess & Index")
    st.info(f"Available reciters in API library: **{len(RECITERS_MAP)}**. Select which ones to train.")
    
    # Selection Tools
    all_codes = sorted(RECITERS_MAP.keys())
    
    col_a, col_b = st.columns([3, 1])
    with col_b:
        st.write("**Selection Controls**")
        if st.button("✅ Select ALL"):
            st.session_state['selected_reciters'] = all_codes
        if st.button("❌ Clear Selection"):
            st.session_state['selected_reciters'] = []
            
    with col_a:
        # Prevent Streamlit exceptions if old session state keys no longer match new API options
        valid_defaults = []
        if 'selected_reciters' in st.session_state:
            valid_defaults = [x for x in st.session_state['selected_reciters'] if x in all_codes]
        else:
            valid_defaults = all_codes[:2] if len(all_codes) >= 2 else all_codes
            
        st.session_state['selected_reciters'] = valid_defaults
            
        selected = st.multiselect("Select Reciters", all_codes, 
                                  default=st.session_state['selected_reciters'], 
                                  key='reciter_selector',
                                  format_func=lambda x: f"{RECITERS_MAP[x]}")
        st.session_state['selected_reciters'] = selected

    st.write(f"**Selected:** {len(selected)} reciters | **Est. Download:** {len(selected)*35} MB")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Start Master Pipeline"):
            codes_str = " ".join(selected)
            cmd = f"nohup {BASE_DIR}/venv/bin/python3 {BASE_DIR}/src/pipeline.py {codes_str} > {BASE_DIR}/logs/pipeline.log 2>&1 &"
            subprocess.Popen(cmd, shell=True)
            st.success("Master Pipeline triggered for " + str(len(selected)) + " reciters!")
            time.sleep(1); st.rerun()
    with c2:
        if st.button("🛑 Force Stop All"):
            subprocess.run(['pkill', '-f', 'pipeline.py'])
            subprocess.run(['pkill', '-f', 'embeddings.py'])
            st.warning("All background tasks killed."); time.sleep(1); st.rerun()
    with c3:
        if st.button("🔄 Refresh Status"): st.rerun()

    st.divider()
    status = load_json(STATUS_FILE)
    if status.get('status') == 'running':
        st.markdown(f"### ⚡ **{status.get('step', 'Processing')}**: {status.get('current_reciter', 'Initializing...')}")
        if status.get('reciter_idx'):
            st.write(f"Reciter {status.get('reciter_idx')} of {status.get('total_reciters')}")
            pct = status.get('reciter_idx') / status.get('total_reciters')
            st.progress(min(pct, 1.0))
        st.info(status.get('message', 'Working...'))
    elif status.get('status') == 'done':
        st.success(f"✅ Last pipeline finished at {status.get('finished_at')}")

    st.subheader("📁 Global Reciter Library Status")
    lib_data = []
    meta = load_json(META_PATH)
    for code, name in RECITERS_MAP.items():
        is_raw = os.path.exists(os.path.join(RAW_DIR, name))
        is_proc = os.path.exists(os.path.join(PROCESSED_DIR, name))
        in_index = name in meta.get('reciters', [])
        lib_data.append({"Reciter": name, "Code": code, "Downloaded": "✅" if is_raw else "❌", "Processed": "✅" if is_proc else "❌", "Indexed": "✅" if in_index else "❌"})
    st.dataframe(pd.DataFrame(lib_data), use_container_width=True, height=500)

elif page == "Identify (Test)":
    st.subheader("🔍 Real-time Reciter Identification")
    uploaded = st.file_uploader("Upload audio...", type=['mp3', 'wav', 'm4a'])
    if uploaded:
        st.audio(uploaded)
        if st.button("🔮 Run AI Identification"):
            with st.spinner("Searching index..."):
                temp = os.path.join(BASE_DIR, "data/temp_test.wav")
                with open(temp, "wb") as f: f.write(uploaded.getbuffer())
                import sys
                sys.path.insert(0, os.path.join(BASE_DIR, "src"))
                from embeddings import EmbeddingExtractor
                ext = EmbeddingExtractor()
                results = ext.identify(temp)
                if results:
                    st.success(f"Best Match: **{results[0]['reciter']}** ({results[0]['similarity']:.4f})")
                    st.table(pd.DataFrame(results)[['reciter', 'similarity']])
                else: st.error("Identification failed. Index may be missing.")

elif page == "System Settings":
    st.subheader("⚙️ OCI Server Resources")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("CPU Load", f"{psutil.cpu_percent(interval=1)}%")
    ram = psutil.virtual_memory()
    sc2.metric("RAM", f"{ram.percent}%", f"{ram.used/1024**3:.1f}G/{ram.total/1024**3:.1f}G")
    disk = psutil.disk_usage('/')
    sc3.metric("Disk", f"{disk.percent}%", f"{disk.free/1024**3:.1f}G Free")
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if any(x in str(p.info['cmdline']) for x in ['python', 'streamlit', 'pipeline']):
                procs.append({'PID': p.info['pid'], 'Name': p.info['name'], 'Cmd': " ".join(p.info['cmdline'][:5])})
        except: pass
    st.table(pd.DataFrame(procs))

st.sidebar.divider()
st.sidebar.caption(f"Server Time: {time.strftime('%H:%M:%S UTC')}")
time.sleep(10); st.rerun()
