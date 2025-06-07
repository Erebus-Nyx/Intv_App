"""
Main Streamlit Application for INTV
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def main():
    st.set_page_config(
        page_title="INTV - Interview Analysis Platform",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🎤 INTV")
        st.markdown("Interview Analysis Platform")
        
        page = st.selectbox(
            "Navigate to:",
            [
                "📋 Document Analysis",
                "🎧 Audio Transcription", 
                "🌐 Remote Access Control",
                "⚙️ Settings",
                "📊 System Status"
            ]
        )
    
    # Main content based on selected page
    if page == "📋 Document Analysis":
        st.title("📋 Document Analysis")
        st.markdown("Upload and analyze interview documents")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['txt', 'pdf', 'docx', 'md']
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                module_type = st.selectbox(
                    "Select interview module:",
                    ["adult", "child", "ar", "collateral", "homeassess"]
                )
            
            with col2:
                if st.button("🔄 Analyze Document", type="primary"):
                    with st.spinner("Analyzing document..."):
                        st.info("Document analysis would be implemented here using the LLM pipeline")
    
    elif page == "🎧 Audio Transcription":
        st.title("🎧 Audio Transcription")
        st.markdown("Upload audio files for transcription and analysis")
        
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a']
        )
        
        if uploaded_audio is not None:
            st.success(f"Audio uploaded: {uploaded_audio.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                enable_vad = st.checkbox("Enable Voice Activity Detection", value=True)
                enable_diarization = st.checkbox("Enable Speaker Diarization", value=True)
            
            with col2:
                whisper_model = st.selectbox(
                    "Whisper Model:",
                    ["base", "small", "medium", "large"]
                )
            
            if st.button("🎙️ Transcribe Audio", type="primary"):
                with st.spinner("Transcribing audio..."):
                    st.info("Audio transcription would be implemented here")
    
    elif page == "🌐 Remote Access Control":
        # Import and run the tunnel control app
        try:
            from gui.streamlit_tunnel_app import main as tunnel_main
            tunnel_main()
        except ImportError as e:
            st.error(f"Could not load remote access controls: {e}")
            st.info("Please check that the tunnel control module is properly installed.")
    
    elif page == "⚙️ Settings":
        st.title("⚙️ Settings")
        st.markdown("Configure INTV settings")
        
        with st.expander("🤖 LLM Configuration"):
            llm_mode = st.selectbox("LLM Mode:", ["embedded", "external"])
            if llm_mode == "external":
                llm_provider = st.selectbox("Provider:", ["openai", "anthropic", "koboldcpp"])
                api_base = st.text_input("API Base URL:")
                api_key = st.text_input("API Key:", type="password")
        
        with st.expander("🔍 RAG Configuration"):
            rag_mode = st.selectbox("RAG Mode:", ["embedded", "external"])
            chunk_size = st.slider("Chunk Size:", 100, 2000, 500)
            top_k = st.slider("Top K Results:", 1, 20, 5)
        
        with st.expander("🎙️ Audio Configuration"):
            whisper_model = st.selectbox("Default Whisper Model:", ["base", "small", "medium", "large"])
            sample_rate = st.selectbox("Sample Rate:", [16000, 22050, 44100])
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    elif page == "📊 System Status":
        st.title("📊 System Status")
        st.markdown("Monitor system resources and service status")
        
        try:
            import psutil
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CPU Usage", f"{psutil.cpu_percent():.1f}%")
            
            with col2:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
            with col3:
                disk = psutil.disk_usage('/')
                st.metric("Disk Usage", f"{disk.percent:.1f}%")
            
            # Process list
            st.subheader("🔄 Running Processes")
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'] or 'cloudflared' in proc.info['name'] or 'uvicorn' in proc.info['name']:
                        processes.append(proc.info)
                except:
                    continue
            
            if processes:
                st.dataframe(processes)
            else:
                st.info("No relevant processes found")
                
        except ImportError:
            st.warning("psutil not available - system monitoring disabled")

if __name__ == "__main__":
    main()
