"""
AI Research Graph Navigator - Main Streamlit Application
"""
import warnings

# Suppress compatibility warnings (Python 3.14, Pydantic, etc.) - MUST be before any other imports
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*BaseSettings.*pydantic-settings.*")
warnings.filterwarnings("ignore", message=".*Pydantic V1.*Python 3.14.*")
warnings.filterwarnings("ignore", category=UserWarning, module="confection")
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config_loader import ConfigLoader

# Import our modules
from src.data_ingestion import ArXivIngester
from src.graph_builder import KnowledgeGraphBuilder
from src.graph_rag import GraphRAGPipeline
from src.graph_statistics import GraphStatistics
from src.improved_chunking import FixedChunker, SemanticChunker
from src.improved_graph_builder import ImprovedKnowledgeGraphBuilder
from src.llm_generator import LLMGenerator
from src.rag_pipeline import RAGPipeline

# Optional imports (app will work without these)
try:
    from src.model_training import ModelTrainer
except ImportError:
    ModelTrainer = None

try:
    from src.evaluation import RAGEvaluator
except ImportError as e:
    print(f"Warning: Evaluation module not available: {e}")
    RAGEvaluator = None

# Page config
st.set_page_config(
    page_title="AI Research Graph Navigator | Enterprise RAG Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI Research Graph Navigator - Enterprise-grade RAG and Graph RAG platform for research paper analysis"
    }
)

# Professional CSS - Modern Berlin Tech Aesthetic
st.markdown("""
<style>
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #0066FF;
        --primary-dark: #0052CC;
        --secondary-color: #6366F1;
        --accent-color: #8B5CF6;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --bg-primary: #FFFFFF;
        --bg-secondary: #F8FAFC;
        --bg-tertiary: #F1F5F9;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-tertiary: #94A3B8;
        --border-color: #E2E8F0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Professional header */
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0066FF 0%, #6366F1 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
        letter-spacing: 0.01em;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: var(--bg-secondary);
    }
    
    /* Section headers */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary);
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    h2 {
        font-size: 1.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        color: var(--text-primary);
    }
    
    /* Professional buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-sm);
        letter-spacing: 0.01em;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }
    
    /* Enhanced metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: var(--text-primary);
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: var(--text-secondary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-container {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .metric-container:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }
    
    /* Cards and containers */
    .stDataFrame {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 0.5rem 0.75rem;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.1);
    }
    
    /* Sliders */
    .stSlider>div>div {
        background-color: var(--bg-tertiary);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: var(--radius-md);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #ECFDF5;
        border-left: 4px solid var(--success-color);
        border-radius: var(--radius-md);
        padding: 1rem;
    }
    
    .stError {
        background-color: #FEF2F2;
        border-left: 4px solid var(--error-color);
        border-radius: var(--radius-md);
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #FFFBEB;
        border-left: 4px solid var(--warning-color);
        border-radius: var(--radius-md);
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #EFF6FF;
        border-left: 4px solid var(--primary-color);
        border-radius: var(--radius-md);
        padding: 1rem;
    }
    
    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 2rem 0;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--text-primary);
        border-radius: var(--radius-sm);
    }
    
    /* Chat interface */
    .stChatMessage {
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--text-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* Professional badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-primary {
        background-color: #DBEAFE;
        color: var(--primary-color);
    }
    
    .badge-success {
        background-color: #D1FAE5;
        color: var(--success-color);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-active {
        background-color: var(--success-color);
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }
    
    .status-inactive {
        background-color: var(--text-tertiary);
    }
    
    /* Professional spacing */
    .section-spacing {
        margin-top: 3rem;
        margin-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration (cached)"""
    return ConfigLoader()

config = load_config()

# Initialize session state
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'config' not in st.session_state:
    st.session_state.config = config
if 'graph_builder' not in st.session_state:
    # Use improved builder if configured
    use_improved = config.get('graph.use_improved_builder', True)
    if use_improved:
        graph_config = config.get_section('graph')
        st.session_state.graph_builder = ImprovedKnowledgeGraphBuilder(
            min_entity_confidence=graph_config.get('min_entity_confidence', 0.5),
            min_edge_weight=graph_config.get('min_edge_weight', 0.3),
            use_pruning=graph_config.get('use_pruning', True)
        )
    else:
        st.session_state.graph_builder = KnowledgeGraphBuilder()
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'graph_rag_pipeline' not in st.session_state:
    st.session_state.graph_rag_pipeline = None
if 'llm_generator' not in st.session_state:
    llm_config = config.get_section('llm')
    st.session_state.llm_generator = LLMGenerator(
        provider=llm_config.get('provider', 'ollama'),
        model_name=llm_config.get('model_name', 'llama3.2'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 200)
    )
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = False
if 'vector_db_ready' not in st.session_state:
    st.session_state.vector_db_ready = False
if 'use_graph_rag' not in st.session_state:
    st.session_state.use_graph_rag = False

# Professional Header with Branding
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1 class="main-header">AI Research Graph Navigator</h1>
    <p class="sub-header">
        Enterprise-grade RAG & Graph RAG Platform for Research Paper Analysis
    </p>
    <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;'>
        <span class="badge badge-primary">RAG Pipeline</span>
        <span class="badge badge-primary">Graph RAG</span>
        <span class="badge badge-primary">Knowledge Graphs</span>
        <span class="badge badge-primary">Vector Search</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Professional Sidebar
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h2 style='margin-bottom: 0.5rem; font-size: 1.5rem;'>⚙️ Configuration</h2>
        <p style='color: var(--text-secondary); font-size: 0.9rem; margin: 0;'>
            Configure and manage your research pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Ingestion Section
    st.markdown("### 📥 Data Ingestion")
    num_papers = st.slider("Number of papers to fetch", 10, 100, 30)
    query = st.text_input("ArXiv Query", value="cat:cs.AI OR cat:cs.LG OR cat:cs.CL")
    
    if st.button("📥 Fetch Papers from ArXiv"):
        with st.spinner("Fetching papers..."):
            ingester = ArXivIngester()
            papers = ingester.fetch_papers(query=query, max_results=num_papers)
            ingester.save_papers(papers)
            st.session_state.papers = papers
            st.success(f"✅ Fetched {len(papers)} papers!")
    
    # Load existing papers
    if st.button("📂 Load Existing Papers"):
        ingester = ArXivIngester()
        papers = ingester.load_papers()
        if papers:
            st.session_state.papers = papers
            st.success(f"✅ Loaded {len(papers)} papers!")
        else:
            st.warning("No papers found. Please fetch papers first.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Build Knowledge Graph Section
    st.markdown("### 🕸️ Knowledge Graph")
    if st.button("🕸️ Build Knowledge Graph"):
        if not st.session_state.papers:
            st.error("Please fetch or load papers first!")
        else:
            try:
                num_papers = len(st.session_state.papers)
                if num_papers > 30:
                    st.info(f"⏳ Processing {num_papers} papers - this may take a few minutes. Please be patient...")
                
                # Use progress tracking for large paper sets
                progress_container = st.empty()
                status_container = st.empty()
                
                if num_papers > 20:
                    # Show progress for large sets - use spinner instead of threading
                    # (Threading doesn't work with Streamlit session_state)
                    graph_file = os.path.join(
                        config.get('data.data_dir', 'data'),
                        config.get('data.graph_file', 'knowledge_graph.json')
                    )
                    
                    # Build graph synchronously with spinner (Streamlit handles this well)
                    with st.spinner(f"Building graph for {num_papers} papers - this may take a few minutes..."):
                        try:
                            st.session_state.graph_builder.build_graph(st.session_state.papers)
                            st.session_state.graph_builder.save_graph(graph_file)
                            st.session_state.graph_built = True
                            stats = st.session_state.graph_builder.get_statistics()
                            st.success(
                                f"✅ Knowledge graph built! "
                                f"({stats.get('nodes', stats.get('total_nodes', 0))} nodes, "
                                f"{stats.get('edges', stats.get('total_edges', 0))} edges)"
                            )
                        except Exception as build_error:
                            st.error(f"Graph building failed: {str(build_error)}")
                            st.exception(build_error)
                else:
                    # Small set - process normally
                    with st.spinner("Building knowledge graph..."):
                        graph_file = os.path.join(
                            config.get('data.data_dir', 'data'),
                            config.get('data.graph_file', 'knowledge_graph.json')
                        )
                        st.session_state.graph_builder.build_graph(st.session_state.papers)
                        st.session_state.graph_builder.save_graph(graph_file)
                        st.session_state.graph_built = True
                        stats = st.session_state.graph_builder.get_statistics()
                        st.success(f"✅ Knowledge graph built! ({stats.get('nodes', stats.get('total_nodes', 0))} nodes, {stats.get('edges', stats.get('total_edges', 0))} edges)")
            except Exception as e:
                st.error(f"Error building graph: {str(e)}")
                st.exception(e)
    
    # Load existing graph
    if st.button("📂 Load Existing Graph"):
        try:
            with st.spinner("Loading graph..."):
                graph_file = os.path.join(
                    config.get('data.data_dir', 'data'),
                    config.get('data.graph_file', 'knowledge_graph.json')
                )
                
                # Check if file exists first
                if not os.path.exists(graph_file):
                    st.warning(f"Graph file not found at: {graph_file}")
                else:
                    # Load graph
                    success = st.session_state.graph_builder.load_graph(graph_file)
                    if success:
                        st.session_state.graph_built = True
                        # Get stats safely
                        try:
                            stats = st.session_state.graph_builder.get_statistics()
                            nodes = stats.get('nodes', stats.get('total_nodes', 0))
                            edges = stats.get('edges', stats.get('total_edges', 0))
                            st.success(f"✅ Graph loaded! ({nodes} nodes, {edges} edges)")
                            st.rerun()  # Refresh to show updated stats
                        except Exception as stats_error:
                            st.success("✅ Graph loaded!")
                            st.warning(f"Could not retrieve statistics: {stats_error}")
                    else:
                        st.warning("Failed to load graph. File may be corrupted.")
        except Exception as e:
            st.error(f"Error loading graph: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize RAG Section
    st.markdown("### 🔍 RAG Pipeline")
    if st.button("🔍 Initialize RAG Pipeline"):
        if not st.session_state.papers:
            st.error("Please fetch papers first!")
        else:
            try:
                # Step 1: Initialize RAG pipeline
                with st.spinner("Step 1/4: Initializing RAG pipeline..."):
                    try:
                        rag_config = config.get_section('rag')
                        chunking_config = config.get_section('chunking')
                        
                        # Initialize RAG pipeline with config
                        st.session_state.rag_pipeline = RAGPipeline(
                            embedding_model=rag_config.get('embedding_model', 'all-MiniLM-L6-v2'),
                            chroma_dir=rag_config.get('chroma_dir', 'chroma_db'),
                            collection_name=rag_config.get('collection_name', 'research_papers')
                        )
                        st.success("✅ RAG pipeline initialized")
                    except ImportError as init_error:
                        if "ChromaDB" in str(init_error) or "chromadb" in str(init_error).lower():
                            st.error(f"ChromaDB import failed: {str(init_error)}")
                            st.info("""
                            **To fix this issue:**
                            1. Install pydantic-settings: `pip install pydantic-settings>=2.0.0`
                            2. Reinstall chromadb: `pip install --upgrade chromadb`
                            3. If using Python 3.14+, ensure all dependencies are compatible
                            
                            Try running: `pip install pydantic-settings>=2.0.0 chromadb`
                            """)
                        else:
                            st.error(f"Failed to initialize RAG pipeline: {str(init_error)}")
                        st.exception(init_error)
                        raise
                    except Exception as init_error:
                        st.error(f"Failed to initialize RAG pipeline: {str(init_error)}")
                        st.exception(init_error)
                        raise
                
                # Step 2: Setup chunker
                try:
                    chunk_size = rag_config.get('chunk_size', 500)
                    chunk_overlap = rag_config.get('chunk_overlap', 50)
                    chunking_strategy = chunking_config.get('strategy', 'semantic')
                    
                    if chunking_strategy == 'semantic':
                        chunker = SemanticChunker(
                            chunk_size=chunk_size,
                            overlap=chunk_overlap,
                            respect_sections=chunking_config.get('respect_sections', True)
                        )
                    else:
                        chunker = FixedChunker(
                            chunk_size=chunk_size,
                            overlap=chunk_overlap
                        )
                except Exception as chunker_error:
                    st.error(f"Failed to create chunker: {str(chunker_error)}")
                    st.exception(chunker_error)
                    raise
                
                # Step 3: Chunk papers (with batch processing for large sets)
                all_chunks = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    num_papers = len(st.session_state.papers)
                    # Use batch processing for 20+ papers to prevent blocking
                    use_batch_processing = num_papers >= 20
                    batch_size = 10 if use_batch_processing else num_papers
                    
                    if use_batch_processing:
                        status_text.text(f"Processing {num_papers} papers in batches of {batch_size}...")
                        # Process in batches with threading for better responsiveness
                        def chunk_paper_safe(paper, index, total):
                            try:
                                if not isinstance(paper, dict):
                                    return None, f"Skipping invalid paper at index {index}"
                                
                                if 'id' not in paper or 'title' not in paper:
                                    return None, f"Skipping paper at index {index} (missing id or title)"
                                
                                chunks = chunker.chunk(paper)
                                return chunks, None
                            except Exception as e:
                                return None, f"Error chunking paper {index+1}: {str(e)}"
                        
                        # Process papers in batches
                        processed = 0
                        for batch_start in range(0, num_papers, batch_size):
                            batch_end = min(batch_start + batch_size, num_papers)
                            batch_papers = st.session_state.papers[batch_start:batch_end]
                            
                            # Process batch with threading (max 4 workers to avoid overwhelming system)
                            with ThreadPoolExecutor(max_workers=4) as executor:
                                futures = {
                                    executor.submit(chunk_paper_safe, paper, batch_start + i, num_papers): 
                                    (batch_start + i, paper) 
                                    for i, paper in enumerate(batch_papers)
                                }
                                
                                for future in as_completed(futures):
                                    index, paper = futures[future]
                                    try:
                                        chunks, error = future.result(timeout=30)  # 30 second timeout per paper
                                        if error:
                                            st.warning(error)
                                        elif chunks:
                                            all_chunks.extend(chunks)
                                        else:
                                            st.warning(f"No chunks created for paper: {paper.get('title', 'Unknown')}")
                                    except Exception as e:
                                        st.warning(f"Timeout or error processing paper {index+1}: {str(e)}")
                                    
                                    processed += 1
                                    progress_bar.progress(processed / num_papers)
                                    status_text.text(
                                        f"Processed {processed}/{num_papers} papers "
                                        f"({len(all_chunks)} chunks so far)..."
                                    )
                    else:
                        # Sequential processing for smaller sets
                        for i, paper in enumerate(st.session_state.papers):
                            try:
                                if not isinstance(paper, dict):
                                    st.warning(f"Skipping invalid paper at index {i}")
                                    continue
                                
                                if 'id' not in paper or 'title' not in paper:
                                    st.warning(f"Skipping paper at index {i} (missing id or title)")
                                    continue
                                
                                status_text.text(f"Chunking paper {i+1}/{num_papers}: {paper.get('title', 'Unknown')[:50]}...")
                                chunks = chunker.chunk(paper)
                                
                                if chunks:
                                    all_chunks.extend(chunks)
                                else:
                                    st.warning(f"No chunks created for paper: {paper.get('title', 'Unknown')}")
                                
                                progress_bar.progress((i + 1) / num_papers)
                            except Exception as paper_error:
                                st.warning(f"Error chunking paper {i+1}: {str(paper_error)}")
                                continue
                    
                    status_text.text(f"✅ Created {len(all_chunks)} chunks from {num_papers} papers")
                    progress_bar.empty()
                except Exception as chunking_error:
                    st.error(f"Error during chunking: {str(chunking_error)}")
                    st.exception(chunking_error)
                    raise
                
                # Step 4: Add documents to vector database (with optimized batching)
                if all_chunks:
                    try:
                        # Use smaller batches for large datasets to prevent memory issues
                        num_chunks = len(all_chunks)
                        if num_chunks > 500:
                            batch_size = 25  # Smaller batches for very large datasets
                        elif num_chunks > 200:
                            batch_size = 50
                        else:
                            batch_size = 50
                        
                        total_batches = (num_chunks + batch_size - 1) // batch_size
                        batch_progress = st.progress(0)
                        batch_status = st.empty()
                        
                        added_count = 0
                        for batch_idx, i in enumerate(range(0, num_chunks, batch_size)):
                            try:
                                batch = all_chunks[i:i+batch_size]
                                batch_status.text(
                                    f"Adding batch {batch_idx+1}/{total_batches} "
                                    f"({len(batch)} chunks, {added_count} total added)..."
                                )
                                
                                # Add batch with timeout protection
                                st.session_state.rag_pipeline.add_documents(batch)
                                added_count += len(batch)
                                
                                batch_progress.progress((batch_idx + 1) / total_batches)
                                
                                # Small delay to keep UI responsive
                                if batch_idx % 5 == 0:
                                    time.sleep(0.1)
                                    
                            except Exception as batch_error:
                                st.warning(f"Error adding batch {batch_idx+1}: {str(batch_error)}")
                                # Try to continue with smaller batch
                                if len(batch) > 1:
                                    try:
                                        # Split into smaller sub-batches
                                        sub_batch_size = len(batch) // 2
                                        for sub_i in range(0, len(batch), sub_batch_size):
                                            sub_batch = batch[sub_i:sub_i+sub_batch_size]
                                            st.session_state.rag_pipeline.add_documents(sub_batch)
                                            added_count += len(sub_batch)
                                    except:
                                        pass
                                continue  # Continue with next batch
                        
                        batch_progress.empty()
                        batch_status.empty()
                        st.success(f"✅ Added {added_count} chunks to vector database")
                    except Exception as add_error:
                        st.error(f"Error adding documents: {str(add_error)}")
                        st.exception(add_error)
                        raise
                
                # Step 5: Initialize Graph RAG if available (only if chunks were created)
                if all_chunks:
                    try:
                        st.session_state.vector_db_ready = True
                        
                        if st.session_state.graph_built:
                            try:
                                st.session_state.graph_rag_pipeline = GraphRAGPipeline(
                                    st.session_state.graph_builder,
                                    st.session_state.rag_pipeline
                                )
                                st.info("✅ Graph RAG pipeline initialized")
                            except Exception as graph_rag_error:
                                st.warning(f"Could not initialize Graph RAG: {str(graph_rag_error)}")
                                st.session_state.graph_rag_pipeline = None
                        else:
                            st.session_state.graph_rag_pipeline = None
                            st.info("💡 Build knowledge graph to enable Graph RAG features")
                    except Exception as final_error:
                        st.warning(f"Error in final setup: {str(final_error)}")
                    
                    # Get final stats
                    try:
                        stats = st.session_state.rag_pipeline.get_collection_stats()
                        st.success(f"✅ RAG pipeline ready! ({stats['total_chunks']} chunks)")
                    except Exception as stats_error:
                        st.success("✅ RAG pipeline ready!")
                        st.warning(f"Could not retrieve statistics: {str(stats_error)}")
                else:
                    st.warning("⚠️ No chunks were created. Please check your papers.")
                    
            except Exception as e:
                st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
                st.exception(e)
                import traceback
                st.code(traceback.format_exc())
                # Don't crash - keep the server running
                st.info("💡 The server is still running. Please check the error above and try again.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # RAG Settings Section
    st.markdown("### ⚙️ RAG Settings")
    st.session_state.use_graph_rag = st.checkbox(
        "Use Graph RAG (enhanced retrieval)",
        value=st.session_state.use_graph_rag,
        help="Enable Graph RAG for enhanced context retrieval through knowledge graph traversal"
    )
    if st.session_state.use_graph_rag and not st.session_state.graph_built:
        st.warning("⚠️ Graph not built. Graph RAG will not be available.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Training Section
    st.markdown("### Model Training")
    if ModelTrainer is None:
        st.warning("Model training module not available. Install required dependencies (transformers, peft, datasets).")
    else:
        if st.button("Start Training"):
            if not st.session_state.papers:
                st.error("Please fetch papers first!")
            else:
                try:
                    training_config = config.get_section('training')
                    
                    with st.spinner("Initializing model trainer..."):
                        trainer = ModelTrainer(
                            base_model=training_config.get('base_model', 'microsoft/Phi-2'),
                            output_dir=training_config.get('output_dir', 'models/fine-tuned')
                        )
                    
                    # Show training parameters
                    st.info(f"""
                    **Training Configuration:**
                    - Model: {training_config.get('base_model', 'microsoft/Phi-2')}
                    - Epochs: {training_config.get('num_epochs', 3)}
                    - Batch Size: {training_config.get('batch_size', 2)}
                    - Learning Rate: {training_config.get('learning_rate', 2e-4)}
                    - LoRA Rank: {training_config.get('lora_r', 16)}
                    """)
                    
                    # Prepare data
                    with st.spinner("Preparing training data from papers..."):
                        train_dataset = trainer.prepare_training_data(st.session_state.papers)
                        st.success(f"Prepared training dataset with {len(train_dataset)} examples")
                    
                    # Training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def training_callback(epoch, logs=None):
                        progress = (epoch + 1) / training_config.get('num_epochs', 3)
                        progress_bar.progress(progress)
                        if logs:
                            status_text.text(f"Epoch {epoch + 1}/{training_config.get('num_epochs', 3)} - Loss: {logs.get('loss', 'N/A'):.4f}")
                    
                    # Start training
                    status_text.text("Starting training... This may take a while (especially for larger models).")
                    
                    with st.spinner("Training model... This may take 10-30 minutes depending on model size and data."):
                        trainer.train(
                            train_dataset=train_dataset,
                            num_epochs=training_config.get('num_epochs', 3),
                            batch_size=training_config.get('batch_size', 2),
                            learning_rate=training_config.get('learning_rate', 2e-4)
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Training complete! Saving model...")
                    
                    # Show results
                    model_path = training_config.get('output_dir', 'models/fine-tuned')
                    
                    # Check if model was actually saved
                    import os
                    model_exists = os.path.exists(model_path) and os.path.isdir(model_path)
                    
                    if model_exists:
                        st.balloons()  # Celebration animation
                        st.success("**Training Complete Successfully!**")
                    else:
                        st.warning("Training completed but model directory not found. Check logs for details.")
                    
                    # Display comprehensive results
                    st.markdown("### Training Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Model Information:**")
                        st.info(f"""
                        - **Base Model**: {training_config.get('base_model', 'microsoft/Phi-2')}
                        - **Output Directory**: `{model_path}`
                        - **Training Examples**: {len(train_dataset)}
                        - **Status**: {'Saved successfully' if model_exists else 'Check logs'}
                        """)
                    
                    with col2:
                        st.markdown("**Training Configuration:**")
                        st.info(f"""
                        - **Epochs**: {training_config.get('num_epochs', 3)}
                        - **Batch Size**: {training_config.get('batch_size', 2)}
                        - **Learning Rate**: {training_config.get('learning_rate', 2e-4)}
                        - **LoRA Rank**: {training_config.get('lora_r', 16)}
                        """)
                    
                    st.markdown("### Next Steps")
                    st.markdown("""
                    You can now use this fine-tuned model for:
                    - **Domain-specific text generation**: Better understanding of research paper terminology
                    - **Research paper summarization**: Generate concise summaries
                    - **Domain-adapted question answering**: More accurate answers on research topics
                    
                    **To use the model:**
                    1. Update your `config.yaml` LLM settings to use Hugging Face provider
                    2. Set the model path to: `{model_path}`
                    3. Or use the model programmatically via the ModelTrainer class
                    """)
                    
                    # Show model info in expandable section
                    with st.expander("Detailed Model Information"):
                        st.code(f"""
Model Configuration:
  Base Model: {training_config.get('base_model', 'microsoft/Phi-2')}
  Output Directory: {model_path}
  Model Exists: {model_exists}
  
Training Parameters:
  Training Examples: {len(train_dataset)}
  Epochs: {training_config.get('num_epochs', 3)}
  Batch Size: {training_config.get('batch_size', 2)}
  Learning Rate: {training_config.get('learning_rate', 2e-4)}
  LoRA Rank: {training_config.get('lora_r', 16)}
  LoRA Alpha: {training_config.get('lora_alpha', 32)}
  
Files Location:
  Model files: {model_path}/
  Tokenizer: {model_path}/tokenizer files
  Adapter weights: {model_path}/adapter_model.bin
                        """)
                    
                    # Store trainer in session state for later use
                    st.session_state.model_trainer = trainer
                    st.session_state.trained_model_path = model_path
                    st.session_state.model_trained = True
                    # Auto-load the model after training for immediate testing
                    st.session_state.trained_model_loaded = True
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.exception(e)
                    st.info("Note: Training requires significant computational resources. GPU recommended for larger models.")
    
    # Load and Test Trained Model Section
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Test Trained Model")
        
        if st.session_state.trained_model_path and os.path.exists(st.session_state.trained_model_path):
            # Model Status Card
            model_status_col1, model_status_col2 = st.columns([3, 1])
            
            with model_status_col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 8px; color: white; margin-bottom: 1rem;'>
                    <h4 style='margin: 0 0 5px 0;'>Trained Model Available</h4>
                    <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>
                        Path: <code style='background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px;'>{st.session_state.trained_model_path}</code>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with model_status_col2:
                model_loaded = st.session_state.get('trained_model_loaded', False)
                if model_loaded:
                    st.success("✓ Loaded")
                else:
                    if st.button("🔄 Load Model", use_container_width=True, type="primary"):
                        try:
                            with st.spinner("Loading trained model... This may take a minute."):
                                if 'model_trainer' in st.session_state and st.session_state.model_trainer:
                                    trainer = st.session_state.model_trainer
                                    # Model is already loaded in trainer
                                    st.session_state.trained_model_loaded = True
                                    st.rerun()
                                else:
                                    # Reload trainer
                                    training_config = config.get_section('training')
                                    trainer = ModelTrainer(
                                        base_model=training_config.get('base_model', 'microsoft/Phi-2'),
                                        output_dir=st.session_state.trained_model_path
                                    )
                                    trainer.load_model(st.session_state.trained_model_path)
                                    st.session_state.model_trainer = trainer
                                    st.session_state.trained_model_loaded = True
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load model: {str(e)}")
                            st.exception(e)
            
            # Auto-load model if available but not loaded
            if not st.session_state.get('trained_model_loaded', False) and 'model_trainer' in st.session_state and st.session_state.model_trainer:
                st.session_state.trained_model_loaded = True
            
            # Test the trained model
            if st.session_state.get('trained_model_loaded', False):
                st.divider()
                
                # Quick Test Prompts
                st.markdown("#### Quick Test Prompts")
                quick_prompts = [
                    "What are the main approaches to transformer architectures?",
                    "Explain how neural networks learn from data",
                    "What is the difference between supervised and unsupervised learning?",
                    "Describe the applications of computer vision",
                    "How do large language models handle few-shot learning?"
                ]
                
                prompt_cols = st.columns(5)
                selected_quick_prompt = None
                for i, prompt in enumerate(quick_prompts):
                    with prompt_cols[i]:
                        if st.button(prompt[:30] + "..." if len(prompt) > 30 else prompt, 
                                   key=f"quick_prompt_{i}", use_container_width=True):
                            selected_quick_prompt = prompt
                            st.session_state.test_prompt = prompt
                
                st.divider()
                
                # Main Testing Interface
                st.markdown("#### Model Testing Interface")
                
                # Prompt Input
                default_prompt = st.session_state.get('test_prompt', selected_quick_prompt or "What are the main approaches to transformer architectures?")
                test_prompt = st.text_area(
                    "Enter your test prompt",
                    value=default_prompt,
                    height=120,
                    help="Enter a question or prompt to test your fine-tuned model",
                    key="test_prompt_input"
                )
                
                # Generation Parameters
                param_col1, param_col2, param_col3, param_col4 = st.columns(4)
                
                with param_col1:
                    test_max_length = st.slider(
                        "Max Length (tokens)",
                        min_value=50,
                        max_value=500,
                        value=200,
                        step=50,
                        help="Maximum number of tokens to generate"
                    )
                
                with param_col2:
                    test_temperature = st.slider(
                        "Temperature",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Controls randomness: lower = more focused, higher = more creative"
                    )
                
                with param_col3:
                    test_top_p = st.slider(
                        "Top-p (Nucleus)",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.9,
                        step=0.1,
                        help="Nucleus sampling parameter"
                    )
                
                with param_col4:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    generate_btn = st.button(
                        "🚀 Generate Response",
                        use_container_width=True,
                        type="primary"
                    )
                
                # Generate Response
                if generate_btn:
                    if st.session_state.model_trainer:
                        try:
                            with st.spinner("Generating response with trained model..."):
                                response = st.session_state.model_trainer.generate(
                                    prompt=test_prompt,
                                    max_length=test_max_length,
                                    temperature=test_temperature,
                                    top_p=test_top_p
                                )
                                
                                # Remove prompt from response if included
                                if response.startswith(test_prompt):
                                    response = response[len(test_prompt):].strip()
                                
                                # Save to session for comparison
                                if 'model_test_results' not in st.session_state:
                                    st.session_state.model_test_results = []
                                
                                result_entry = {
                                    'prompt': test_prompt,
                                    'response': response,
                                    'max_length': test_max_length,
                                    'temperature': test_temperature,
                                    'top_p': test_top_p,
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                st.session_state.model_test_results.append(result_entry)
                                st.session_state.last_generation = result_entry
                                
                                st.rerun()
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")
                            st.exception(e)
                    else:
                        st.error("Model trainer not available. Please reload the model.")
                
                # Display Last Generation
                if 'last_generation' in st.session_state and st.session_state.last_generation:
                    st.divider()
                    st.markdown("#### Generated Response")
                    
                    result = st.session_state.last_generation
                    
                    # Response Card
                    st.markdown(f"""
                    <div style='background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 1rem;'>
                        <div style='margin-bottom: 15px;'>
                            <strong style='color: #667eea;'>Prompt:</strong>
                            <p style='margin: 5px 0; color: #333;'>{result['prompt']}</p>
                        </div>
                        <div>
                            <strong style='color: #667eea;'>Response:</strong>
                            <p style='margin: 5px 0; color: #333; white-space: pre-wrap;'>{result['response']}</p>
                        </div>
                        <div style='margin-top: 15px; font-size: 0.85rem; color: #666;'>
                            <span>Max Length: {result['max_length']}</span> | 
                            <span>Temperature: {result['temperature']}</span> | 
                            <span>Time: {result['timestamp']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action Buttons
                    action_col1, action_col2, action_col3 = st.columns(3)
                    with action_col1:
                        if st.button("📋 Copy Response", use_container_width=True):
                            st.write("Response copied! (Use Ctrl+C to copy)")
                    with action_col2:
                        if st.button("🔄 Regenerate", use_container_width=True):
                            st.session_state.test_prompt = result['prompt']
                            st.rerun()
                    with action_col3:
                        if st.button("🗑️ Clear", use_container_width=True):
                            st.session_state.last_generation = None
                            st.rerun()
                
                # Test Results History
                if 'model_test_results' in st.session_state and st.session_state.model_test_results:
                    st.divider()
                    with st.expander(f"📊 Test History ({len(st.session_state.model_test_results)} tests)", expanded=False):
                        for i, result in enumerate(reversed(st.session_state.model_test_results[-10:]), 1):
                            with st.container():
                                st.markdown(f"**Test #{len(st.session_state.model_test_results) - i + 1}** - {result['timestamp']}")
                                st.markdown(f"**Prompt:** {result['prompt'][:100]}{'...' if len(result['prompt']) > 100 else ''}")
                                st.markdown(f"**Response:** {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
                                st.caption(f"Settings: Length={result['max_length']}, Temp={result['temperature']}, Top-p={result.get('top_p', 0.9)}")
                                
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if st.button(f"View Full", key=f"view_{i}"):
                                        st.session_state.last_generation = result
                                        st.rerun()
                                with col2:
                                    if st.button(f"Delete", key=f"delete_{i}"):
                                        st.session_state.model_test_results.pop(len(st.session_state.model_test_results) - i)
                                        st.rerun()
                                
                                if i < len(st.session_state.model_test_results[-10:]):
                                    st.divider()
                        
                        # Export Results
                        if st.button("📥 Export All Results", use_container_width=True):
                            import json
                            results_json = json.dumps(st.session_state.model_test_results, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=results_json,
                                file_name=f"model_test_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                
                # RAG Integration
                st.divider()
                st.markdown("#### RAG Pipeline Integration")
                
                integration_col1, integration_col2 = st.columns([3, 1])
                
                with integration_col1:
                    use_trained_in_rag = st.checkbox(
                        "✅ Use trained model for RAG answers (instead of Ollama/Hugging Face)",
                        value=st.session_state.get('use_trained_in_rag', False),
                        help="When enabled, the RAG chat will use your trained model for answer generation. This provides domain-specific answers based on your fine-tuned model."
                    )
                    st.session_state.use_trained_in_rag = use_trained_in_rag
                
                with integration_col2:
                    if use_trained_in_rag:
                        st.success("Active")
                    else:
                        st.info("Inactive")
                
                if use_trained_in_rag:
                    st.info("💡 **Tip**: Go to the 'RAG Chat' tab to ask questions. Your trained model will generate answers using retrieved context from your research papers.")
        else:
            st.warning("⚠️ Trained model path not found. Please train a model first.")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🔍 RAG Chat", 
    "🕸️ Graph Explorer", 
    "📈 Evaluation", 
    "📚 Papers"
])

# Tab 1: Professional Dashboard
with tab1:
    # Dashboard Header
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>📊 System Dashboard</h1>
        <p style='color: var(--text-secondary); font-size: 1rem;'>
            Real-time overview of your research pipeline and knowledge graph
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Indicators Row
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        status = "active" if st.session_state.papers else "inactive"
        st.markdown(f"""
        <div class="metric-container">
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <span class="status-indicator status-{'active' if st.session_state.papers else 'inactive'}"></span>
                <span style='font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;'>
                    Papers Loaded
                </span>
            </div>
            <div style='font-size: 2rem; font-weight: 700; color: var(--text-primary);'>
                {len(st.session_state.papers)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        graph_status = st.session_state.graph_built
        if graph_status and st.session_state.graph_builder:
            try:
                stats = st.session_state.graph_builder.get_statistics()
                node_count = stats.get('nodes', stats.get('total_nodes', 0))
            except:
                node_count = 0
                graph_status = False
        else:
            node_count = 0
        
        st.markdown(f"""
        <div class="metric-container">
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <span class="status-indicator status-{'active' if graph_status else 'inactive'}"></span>
                <span style='font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;'>
                    Graph Nodes
                </span>
            </div>
            <div style='font-size: 2rem; font-weight: 700; color: var(--text-primary);'>
                {node_count:,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col3:
        graph_status = st.session_state.graph_built
        if graph_status and st.session_state.graph_builder:
            try:
                stats = st.session_state.graph_builder.get_statistics()
                edge_count = stats.get('edges', stats.get('total_edges', 0))
            except:
                edge_count = 0
        else:
            edge_count = 0
        
        st.markdown(f"""
        <div class="metric-container">
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <span class="status-indicator status-{'active' if graph_status else 'inactive'}"></span>
                <span style='font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;'>
                    Graph Edges
                </span>
            </div>
            <div style='font-size: 2rem; font-weight: 700; color: var(--text-primary);'>
                {edge_count:,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col4:
        vector_status = st.session_state.vector_db_ready
        if vector_status and st.session_state.rag_pipeline:
            try:
                stats = st.session_state.rag_pipeline.get_collection_stats()
                chunk_count = stats.get('total_chunks', 0)
            except Exception as e:
                chunk_count = 0
                vector_status = False
        else:
            chunk_count = 0
        
        st.markdown(f"""
        <div class="metric-container">
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <span class="status-indicator status-{'active' if vector_status else 'inactive'}"></span>
                <span style='font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;'>
                    Vector Chunks
                </span>
            </div>
            <div style='font-size: 2rem; font-weight: 700; color: var(--text-primary);'>
                {chunk_count:,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Professional Graph Statistics Dashboard
    if st.session_state.graph_built and st.session_state.graph_builder:
        try:
            # Initialize statistics generator
            try:
                graph_stats = GraphStatistics(st.session_state.graph_builder.graph)
                stats = graph_stats.get_comprehensive_stats()
                viz_data = graph_stats.get_visualization_data()
            except Exception as stats_error:
                # Fallback to basic statistics
                st.warning(f"Using basic statistics (advanced stats unavailable: {stats_error})")
                basic_stats = st.session_state.graph_builder.get_statistics()
                # Convert to comprehensive format
                stats = {
                    "total_nodes": basic_stats.get('nodes', 0),
                    "total_edges": basic_stats.get('edges', 0),
                    "node_types": {
                        "papers": basic_stats.get('papers', 0),
                        "authors": basic_stats.get('authors', 0),
                        "concepts": basic_stats.get('concepts', 0)
                    },
                    "density": 0,
                    "avg_degree": 0,
                    "is_connected": basic_stats.get('is_connected', False),
                    "num_components": 1 if basic_stats.get('is_connected', False) else 0
                }
                node_type_df = pd.DataFrame([
                    {"Type": k.title(), "Count": v}
                    for k, v in stats.get("node_types", {}).items() if v > 0
                ])
                viz_data = {
                    "node_type_distribution": node_type_df,
                    "edge_type_distribution": pd.DataFrame(),
                    "degree_distribution": pd.DataFrame(),
                    "stats": stats
                }
            
            # Main Statistics Header
            st.markdown("""
            <div style='margin-top: 2rem; margin-bottom: 1.5rem;'>
                <h2 style='margin-bottom: 0.5rem;'>📊 Comprehensive Graph Analytics</h2>
                <p style='color: var(--text-secondary); font-size: 0.95rem;'>
                    Detailed insights into your knowledge graph structure and relationships
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Metrics Row - Professional Style
            st.markdown("### 📈 Key Performance Indicators")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            
            with kpi_col1:
                st.metric(
                    label="Total Nodes",
                    value=f"{stats.get('total_nodes', 0):,}",
                    delta=None
                )
            with kpi_col2:
                st.metric(
                    label="Total Edges",
                    value=f"{stats.get('total_edges', 0):,}",
                    delta=None
                )
            with kpi_col3:
                st.metric(
                    label="Graph Density",
                    value=f"{stats.get('density', 0):.4f}",
                    delta=None
                )
            with kpi_col4:
                st.metric(
                    label="Avg Degree",
                    value=f"{stats.get('avg_degree', 0):.2f}",
                    delta=None
                )
            with kpi_col5:
                st.metric(
                    label="Components",
                    value=f"{stats.get('num_components', 0)}",
                    delta=None
                )
            
            st.divider()
            
            # Visualization Section
            st.markdown("### 📊 Visual Analytics")
            
            # Row 1: Node and Edge Distributions
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("#### Node Type Distribution")
                node_type_df = viz_data["node_type_distribution"]
                if not node_type_df.empty:
                    fig_pie = px.pie(
                        node_type_df,
                        values='Count',
                        names='Type',
                        title="",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4
                    )
                    fig_pie.update_layout(
                        showlegend=True,
                        height=400,
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No node type data available")
            
            with viz_col2:
                st.markdown("#### Relationship Type Distribution")
                edge_type_df = viz_data["edge_type_distribution"]
                if not edge_type_df.empty:
                    fig_bar = px.bar(
                        edge_type_df.head(10),
                        x='Relation',
                        y='Count',
                        title="",
                        color='Count',
                        color_continuous_scale='Viridis',
                        text='Count'
                    )
                    fig_bar.update_layout(
                        xaxis_tickangle=-45,
                        height=400,
                        showlegend=False,
                        font=dict(size=12)
                    )
                    fig_bar.update_traces(textposition='outside')
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No edge type data available")
            
            # Row 2: Network Metrics
            st.markdown("#### Network Topology Metrics")
            network_col1, network_col2, network_col3 = st.columns(3)
            
            with network_col1:
                metric_card = f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                    <h3 style='margin: 0; font-size: 14px; opacity: 0.9;'>Clustering Coefficient</h3>
                    <h2 style='margin: 10px 0; font-size: 32px;'>{stats.get('avg_clustering', 0):.4f}</h2>
                </div>
                """
                st.markdown(metric_card, unsafe_allow_html=True)
            
            with network_col2:
                metric_card = f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                    <h3 style='margin: 0; font-size: 14px; opacity: 0.9;'>Average Path Length</h3>
                    <h2 style='margin: 10px 0; font-size: 32px;'>{stats.get('avg_path_length', 0):.2f}</h2>
                </div>
                """
                st.markdown(metric_card, unsafe_allow_html=True)
            
            with network_col3:
                connectivity_status = "✅ Connected" if stats.get('is_connected', False) else "⚠️ Disconnected"
                metric_card = f"""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                    <h3 style='margin: 0; font-size: 14px; opacity: 0.9;'>Connectivity Status</h3>
                    <h2 style='margin: 10px 0; font-size: 24px;'>{connectivity_status}</h2>
                </div>
                """
                st.markdown(metric_card, unsafe_allow_html=True)
            
            st.divider()
            
            # Detailed Statistics Tables
            st.markdown("### 📋 Detailed Statistics")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("#### Node Type Breakdown")
                node_types = stats.get('node_types', {})
                if node_types:
                    node_breakdown = pd.DataFrame([
                        {"Node Type": k, "Count": v, "Percentage": f"{(v/stats.get('total_nodes', 1)*100):.1f}%"}
                        for k, v in sorted(node_types.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(
                        node_breakdown,
                        use_container_width=True,
                        hide_index=True,
                        height=300
                    )
                else:
                    st.info("No node type data available")
            
            with detail_col2:
                st.markdown("#### Top Relations")
                top_relations = stats.get('top_relations', [])
                if top_relations:
                    relations_df = pd.DataFrame([
                        {"Relation": k, "Count": v}
                        for k, v in top_relations[:10]
                    ])
                    st.dataframe(
                        relations_df,
                        use_container_width=True,
                        hide_index=True,
                        height=300
                    )
                else:
                    st.info("No relation data available")
            
            # Advanced Metrics Table
            st.markdown("#### Advanced Network Metrics")
            advanced_metrics = {
                "Metric": [
                    "Total Nodes", "Total Edges", "Graph Density", "Average Degree",
                    "Average Clustering", "Number of Components", "Largest Component Size",
                    "Average Path Length", "Diameter"
                ],
                "Value": [
                    f"{stats.get('total_nodes', 0):,}",
                    f"{stats.get('total_edges', 0):,}",
                    f"{stats.get('density', 0):.6f}",
                    f"{stats.get('avg_degree', 0):.2f}",
                    f"{stats.get('avg_clustering', 0):.4f}",
                    f"{stats.get('num_components', 0)}",
                    f"{stats.get('largest_component_size', 0)}",
                    f"{stats.get('avg_path_length', 0):.2f}" if stats.get('avg_path_length', 0) > 0 else "N/A (disconnected)",
                    f"{stats.get('diameter', 0)}" if stats.get('diameter', 0) > 0 else "N/A (disconnected)"
                ]
            }
            advanced_df = pd.DataFrame(advanced_metrics)
            st.dataframe(advanced_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error displaying graph statistics: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Tab 2: Professional RAG Chat
with tab2:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>💬 Intelligent Research Assistant</h1>
        <p style='color: var(--text-secondary); font-size: 1rem;'>
            Ask questions about your research papers using advanced RAG and Graph RAG technology
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.vector_db_ready:
        st.warning("⚠️ Please initialize RAG pipeline from the sidebar first!")
    else:
        # Show LLM status
        if st.session_state.llm_generator:
            llm_provider = st.session_state.llm_generator.provider
            if llm_provider == "simplified":
                st.info("ℹ️ **LLM Status**: Ollama/Hugging Face not available. Using simplified text extraction mode. Answers will be based on retrieved context only.")
            elif llm_provider == "ollama":
                st.success("✅ **LLM Status**: Ollama is available and ready for generation.")
            elif llm_provider == "huggingface":
                st.success("✅ **LLM Status**: Hugging Face model loaded and ready.")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        query = st.chat_input("Ask a question about the research papers...")
        
        if query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            # Get response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching papers..."):
                        # Use RAG or Graph RAG based on session state setting
                        use_graph_rag = st.session_state.use_graph_rag
                        
                        if use_graph_rag and st.session_state.graph_rag_pipeline and st.session_state.graph_built:
                            results = st.session_state.graph_rag_pipeline.graph_retrieve(
                                query, 
                                top_k=config.get('rag.top_k', 5)
                            )
                            context = st.session_state.graph_rag_pipeline.format_graph_context(results)
                        else:
                            if use_graph_rag and not st.session_state.graph_built:
                                st.warning("Graph not built. Using standard RAG.")
                            results = st.session_state.rag_pipeline.retrieve(
                                query, 
                                top_k=config.get('rag.top_k', 5)
                            )
                            context = st.session_state.rag_pipeline.format_context(results)
                        
                        # Generate prompt
                        prompt = st.session_state.rag_pipeline.generate_prompt(query, context)
                        
                        # Generate answer using LLM or trained model
                        with st.spinner("Generating answer..."):
                            # Check if trained model should be used
                            if st.session_state.get('use_trained_in_rag', False) and st.session_state.get('trained_model_loaded', False) and st.session_state.get('model_trainer'):
                                try:
                                    # Use trained model for generation
                                    answer = st.session_state.model_trainer.generate(
                                        prompt=prompt,
                                        max_length=config.get('llm.max_tokens', 200),
                                        temperature=config.get('llm.temperature', 0.7),
                                        top_p=0.9
                                    )
                                    # Remove the prompt from the answer if it's included
                                    if answer.startswith(prompt):
                                        answer = answer[len(prompt):].strip()
                                    st.caption("Using trained fine-tuned model for generation")
                                except Exception as e:
                                    st.warning(f"Trained model generation failed: {str(e)}. Falling back to default LLM.")
                                    answer = st.session_state.llm_generator.generate(prompt, context)
                            else:
                                # Use default LLM generator
                                answer = st.session_state.llm_generator.generate(prompt, context)
                                
                                # Show note if using simplified generation (Ollama not available)
                                if st.session_state.llm_generator.provider == "simplified":
                                    st.info("Note: Using simplified generation (Ollama not available). Install Ollama for better answers, or use your trained model.")
                        
                        st.markdown(answer)
                        
                        # Show sources
                        with st.expander("📚 Sources"):
                            for i, result in enumerate(results[:3], 1):
                                title = result.get('metadata', {}).get('title', 'Unknown')
                                distance = result.get('distance', 0)
                                relevance = 1 - distance if distance else 1.0
                                st.markdown(f"**Source {i}:** {title}")
                                st.markdown(f"*Relevance: {relevance:.2f}*")
                                if 'text' in result:
                                    st.markdown(f"*Excerpt: {result['text'][:200]}...*")
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.exception(e)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Tab 3: Professional Graph Explorer
with tab3:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>🕸️ Knowledge Graph Explorer</h1>
        <p style='color: var(--text-secondary); font-size: 1rem;'>
            Explore relationships and connections in your research knowledge graph
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.graph_built:
        st.warning("⚠️ Please build knowledge graph from the sidebar first!")
    else:
        # Graph visualization options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Options")
            max_nodes = st.slider("Max Nodes", 50, 500, 100)
            layout = st.selectbox("Layout", ["spring", "circular", "kamada_kawai"])
            show_labels = st.checkbox("Show Labels", value=True)
        
        with col1:
            # Get subgraph for visualization
            if st.session_state.papers and st.session_state.graph_built:
                try:
                    # Get sample papers
                    sample_papers = [p['id'] for p in st.session_state.papers[:10]]
                    subgraph = st.session_state.graph_builder.get_subgraph(sample_papers, depth=1)
                    
                    # Create networkx graph for plotly
                    if layout == "spring":
                        pos = nx.spring_layout(subgraph, k=1, iterations=50)
                    elif layout == "circular":
                        pos = nx.circular_layout(subgraph)
                    elif layout == "kamada_kawai":
                        try:
                            pos = nx.kamada_kawai_layout(subgraph)
                        except:
                            pos = nx.spring_layout(subgraph)
                    else:
                        pos = nx.spring_layout(subgraph)
                    
                    # Extract node and edge data
                    edge_x = []
                    edge_y = []
                    for edge in subgraph.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    # Create edge trace
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    # Node traces by type
                    node_traces = []
                    node_types = {}
                    for node in subgraph.nodes():
                        node_data = subgraph.nodes[node]
                        node_type = node_data.get('node_type', 'unknown')
                        if node_type not in node_types:
                            node_types[node_type] = {'x': [], 'y': [], 'text': [], 'ids': []}
                        x, y = pos[node]
                        node_types[node_type]['x'].append(x)
                        node_types[node_type]['y'].append(y)
                        title = node_data.get('title', node_data.get('name', node))
                        node_types[node_type]['text'].append(title[:50])
                        node_types[node_type]['ids'].append(node)
                    
                    # Color mapping
                    colors = {
                        'paper': '#FF6B6B',
                        'author': '#4ECDC4',
                        'concept': '#95E1D3',
                        'category': '#F38181',
                        'keyword': '#AA96DA'
                    }
                    
                    for node_type, data in node_types.items():
                        trace = go.Scatter(
                            x=data['x'],
                            y=data['y'],
                            mode='markers+text' if show_labels else 'markers',
                            name=node_type,
                            text=data['text'] if show_labels else [],
                            textposition="middle right",
                            hovertext=data['ids'],
                            marker=dict(
                                size=10,
                                color=colors.get(node_type, '#888'),
                                line=dict(width=2, color='white')
                            )
                        )
                        node_traces.append(trace)
                    
                    # Create figure
                    fig = go.Figure(
                        data=[edge_trace] + node_traces,
                        layout=go.Layout(
                            title='Knowledge Graph Visualization',
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error visualizing graph: {str(e)}")
                    st.info("Try building the graph again or reducing the number of nodes.")
        
        # Paper exploration
        st.subheader("🔍 Explore Papers")
        if st.session_state.papers:
            paper_titles = [p['title'] for p in st.session_state.papers]
            selected_paper = st.selectbox("Select a paper", paper_titles)
            
            if selected_paper:
                try:
                    paper_id = next(p['id'] for p in st.session_state.papers if p['title'] == selected_paper)
                    neighbors = st.session_state.graph_builder.get_paper_neighbors(paper_id)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Related Papers:**")
                        related_papers = [n for n in neighbors if n['type'] == 'paper']
                        for paper in related_papers[:5]:
                            st.markdown(f"- {paper['data'].get('title', paper['id'])}")
                    
                    with col2:
                        st.markdown("**Key Concepts:**")
                        concepts = [n for n in neighbors if n['type'] == 'concept']
                        for concept in concepts[:5]:
                            st.markdown(f"- {concept['data'].get('name', concept['id'])}")
                except Exception as e:
                    st.error(f"Error exploring paper: {str(e)}")

# Tab 4: Professional Evaluation
with tab4:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>📈 Performance Evaluation</h1>
        <p style='color: var(--text-secondary); font-size: 1rem;'>
            Evaluate RAG pipeline performance using RAGAS metrics and benchmarks
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.vector_db_ready:
        st.warning("⚠️ Please initialize RAG pipeline first!")
    else:
        st.subheader("RAGAS Evaluation Metrics")
        
        # Sample questions for evaluation
        sample_questions = [
            "What are the main approaches to transformer architectures?",
            "How do large language models handle few-shot learning?",
            "What is the difference between supervised and unsupervised learning?",
            "How do neural networks learn from data?",
            "What are the applications of computer vision?"
        ]
        
        questions = st.text_area(
            "Enter questions (one per line)",
            value="\n".join(sample_questions),
            height=150
        )
        
        if st.button("📊 Run Evaluation"):
            if RAGEvaluator is None:
                st.error("Evaluation module not available. Please install required dependencies (ragas, pydantic).")
            else:
                question_list = [q.strip() for q in questions.split("\n") if q.strip()]
                
                if question_list:
                    with st.spinner("Running evaluation..."):
                        try:
                            evaluator = RAGEvaluator(
                                st.session_state.rag_pipeline,
                                st.session_state.graph_rag_pipeline
                            )
                            
                            # Get retrieval stats
                            stats = evaluator.get_retrieval_stats(question_list)
                            
                            # Create evaluation dataset
                            eval_dataset = evaluator.create_evaluation_dataset(question_list)
                            
                            # Run evaluation
                            metrics = evaluator.evaluate_rag(eval_dataset)
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.3f}")
                            
                            with col2:
                                st.metric("Answer Relevancy", f"{metrics.get('answer_relevancy', 0):.3f}")
                            
                            with col3:
                                st.metric("Context Precision", f"{metrics.get('context_precision', 0):.3f}")
                            
                            with col4:
                                st.metric("Context Recall", f"{metrics.get('context_recall', 0):.3f}")
                            
                            # Metrics chart
                            metrics_df = pd.DataFrame([
                                {"Metric": "Faithfulness", "Score": metrics.get('faithfulness', 0)},
                                {"Metric": "Answer Relevancy", "Score": metrics.get('answer_relevancy', 0)},
                                {"Metric": "Context Precision", "Score": metrics.get('context_precision', 0)},
                                {"Metric": "Context Recall", "Score": metrics.get('context_recall', 0)}
                            ])
                            
                            fig = px.bar(
                                metrics_df,
                                x="Metric",
                                y="Score",
                                title="RAGAS Evaluation Metrics",
                                color="Score",
                                color_continuous_scale="Viridis"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Retrieval stats
                            st.subheader("📊 Retrieval Statistics")
                            stats_df = pd.DataFrame([stats])
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.error(f"Error running evaluation: {str(e)}")
                            st.exception(e)

# Tab 5: Professional Papers Browser
with tab5:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>📚 Research Papers Library</h1>
        <p style='color: var(--text-secondary); font-size: 1rem;'>
            Browse and search through your collection of research papers
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.papers:
        st.warning("⚠️ Please fetch papers from the sidebar first!")
    else:
        # Search and filter
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search papers", placeholder="Search by title, author, or category...")
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Title", "Published Date", "Category"])
        
        # Filter papers
        filtered_papers = st.session_state.papers
        if search_query:
            filtered_papers = [
                p for p in filtered_papers
                if search_query.lower() in p['title'].lower() or
                   any(search_query.lower() in str(auth).lower() for auth in p.get('authors', [])) or
                   any(search_query.lower() in str(cat).lower() for cat in p.get('categories', []))
            ]
        
        # Sort papers
        if sort_by == "Title":
            filtered_papers.sort(key=lambda x: x['title'])
        elif sort_by == "Published Date":
            filtered_papers.sort(key=lambda x: x.get('published', ''), reverse=True)
        
        st.info(f"Showing {len(filtered_papers)} of {len(st.session_state.papers)} papers")
        
        # Display papers
        for i, paper in enumerate(filtered_papers):
            with st.expander(f"📄 {paper['title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Authors:** {', '.join(paper.get('authors', [])[:5])}")
                    st.markdown(f"**Published:** {paper.get('published', 'N/A')}")
                    st.markdown(f"**Categories:** {', '.join(paper.get('categories', []))}")
                    st.markdown(f"**Summary:** {paper.get('summary', '')[:500]}...")
                
                with col2:
                    st.markdown(f"**ID:** {paper['id']}")
                    if paper.get('pdf_url'):
                        st.markdown(f"[📥 PDF]({paper['pdf_url']})")
                    if paper.get('url'):
                        st.markdown(f"[🔗 ArXiv]({paper['url']})")

# Professional Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style='text-align: center; color: var(--text-secondary); padding: 2rem 1rem;'>
    <p style='font-size: 0.95rem; margin-bottom: 0.5rem; font-weight: 500;'>
        <strong>AI Research Graph Navigator</strong> — Enterprise RAG & Graph RAG Platform
    </p>
    <p style='font-size: 0.85rem; color: var(--text-tertiary); margin: 0;'>
        Built with advanced RAG, Graph RAG, Knowledge Graphs, and Vector Search technologies
    </p>
    <p style='font-size: 0.75rem; color: var(--text-tertiary); margin-top: 1rem;'>
        © 2024 — Production-Ready Research Analysis Platform
    </p>
</div>
""", unsafe_allow_html=True)

