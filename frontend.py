import streamlit as st
import requests
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="RAG Coding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Custom CSS for developer-friendly UI with chat interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat interface styles */
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        background: rgba(102, 126, 234, 0.02);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.75rem 0;
        margin-left: auto;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: rgba(102, 126, 234, 0.08);
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.75rem 0;
        max-width: 85%;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .chat-container {
            background: rgba(255, 255, 255, 0.02);
        }
        .assistant-message {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(102, 126, 234, 0.3);
        }
    }
    
    [data-theme="dark"] .chat-container {
        background: rgba(255, 255, 255, 0.02);
    }
    
    [data-theme="dark"] .assistant-message {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Source cards that work in both light and dark mode */
    .source-card {
        background: var(--background-color);
        border: 2px solid #667eea;
        border-left: 6px solid #667eea;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .source-header {
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text-color);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .source-meta {
        font-size: 0.95rem;
        margin-bottom: 1rem;
        color: var(--text-color);
        opacity: 0.8;
    }
    
    .relevance-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
    
    .source-excerpt {
        background: rgba(102, 126, 234, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        color: var(--text-color);
        overflow-x: auto;
    }
    
    /* Ensure good contrast in both modes */
    @media (prefers-color-scheme: dark) {
        .source-card {
            background: rgba(255, 255, 255, 0.05);
        }
        .source-excerpt {
            background: rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.3);
        }
    }
    
    @media (prefers-color-scheme: light) {
        .source-card {
            background: rgba(102, 126, 234, 0.02);
        }
    }
    
    /* Streamlit dark mode override */
    [data-theme="dark"] .source-card {
        background: rgba(255, 255, 255, 0.05);
    }
    
    [data-theme="dark"] .source-excerpt {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 RAG Coding Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered code generation grounded in your documentation</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Number of source chunks", min_value=3, max_value=10, value=5, 
                      help="How many relevant document chunks to retrieve")
    
    st.markdown("---")
    
    # Conversation stats
    st.markdown("### 💬 Conversation Stats")
    msg_count = len(st.session_state.chat_messages)
    user_msgs = len([m for m in st.session_state.chat_messages if m['role'] == 'user'])
    assistant_msgs = len([m for m in st.session_state.chat_messages if m['role'] == 'assistant'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", msg_count)
        st.metric("You", user_msgs)
    with col2:
        st.metric("Assistant", assistant_msgs)
    
    if st.session_state.conversation_history:
        st.caption(f"📅 Started: {st.session_state.chat_messages[0]['timestamp'] if st.session_state.chat_messages else 'N/A'}")
    
    st.markdown("---")
    
    st.markdown("### 📖 About")
    st.markdown("""
    This RAG assistant:
    - ✅ Only uses your uploaded docs
    - 📍 Provides exact citations
    - 🚫 Prevents hallucination
    - ⚡ Tracks performance
    - 💬 Remembers conversation context
    """)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["�� Knowledge Base", "🔮 Generate Code", "📊 Analytics"])

# =============================================================================
# TAB 1: Knowledge Base Upload
# =============================================================================
with tab1:
    st.header("📚 Upload Reference Documents")
    st.markdown("Upload your documentation, API references, or code examples to build your knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        accept_multiple_files=True,
        type=['txt', 'md', 'pdf', 'docx'],
        help="Supported formats: TXT, Markdown, PDF, DOCX"
    )
    
    if uploaded_files:
        with st.spinner('🔄 Processing documents...'):
            files = [(f.name, f.read()) for f in uploaded_files]
            response = requests.post(f"{API_URL}/upload", files=[("files", (name, content)) for name, content in files])
        
        if response.ok:
            result = response.json()
            
            # Success message
            if result.get("status") == "success":
                st.success(f"✅ Successfully processed {result['chunks']} chunks from {result['files_processed']} file(s)!")
                
                # Display timing information
                if "timing" in result:
                    timing = result["timing"]
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ Total Time", f"{timing['total_pipeline_time']}s")
                    with col2:
                        st.metric("📦 Chunks", result['chunks'])
                    with col3:
                        st.metric("📄 Files", result['files_processed'])
                    with col4:
                        if timing.get('files'):
                            avg_time = sum(f['total_time'] for f in timing['files'].values()) / len(timing['files'])
                            st.metric("⚡ Avg/File", f"{avg_time:.1f}s")
                    
                    # Per-file breakdown
                    if "files" in timing:
                        st.markdown("### 📊 Detailed Breakdown")
                        for filename, file_timing in timing["files"].items():
                            with st.expander(f"📄 {filename}", expanded=False):
                                # File info
                                info_col1, info_col2, info_col3 = st.columns(3)
                                with info_col1:
                                    st.metric("File Size", f"{file_timing['file_size_mb']:.2f} MB")
                                with info_col2:
                                    st.metric("Pages", file_timing.get('pages_extracted', 'N/A'))
                                with info_col3:
                                    st.metric("Chunks", file_timing['chunks_created'])
                                
                                st.markdown("**⏱️ Time Breakdown:**")
                                time_data = {
                                    "📖 Extract": file_timing['extract_time'],
                                    "✂️ Chunk": file_timing['chunk_time'],
                                    "🧠 Embed": file_timing['embed_time']
                                }
                                
                                for label, time_val in time_data.items():
                                    percentage = (time_val / file_timing['total_time']) * 100
                                    st.progress(percentage / 100, text=f"{label}: {time_val:.2f}s ({percentage:.1f}%)")
                                
                                # Performance metrics
                                st.markdown("**�� Performance:**")
                                st.text(f"• Extract: {file_timing['chars_extracted']/file_timing['extract_time']:.0f} chars/sec")
                                st.text(f"• Embed: {file_timing['chunks_created']/file_timing['embed_time']:.2f} chunks/sec")
                
                # Show progress log in expander
                if "progress" in result:
                    with st.expander("📋 View detailed progress log"):
                        for step in result["progress"]:
                            st.text(step)
            else:
                st.error(f"❌ Upload error: {result.get('message', 'Unknown error')}")
        else:
            st.error(f"❌ Upload failed with status {response.status_code}")

# =============================================================================
# TAB 2: Conversational Code Generation (Chat Interface)
# =============================================================================
with tab2:
    st.header("� Conversational Code Assistant")
    st.markdown("Ask questions, request code modifications, and have natural follow-up conversations - all grounded in your documentation.")
    
    # Chat controls at the top
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🗑️ Clear Conversation", help="Start a new conversation"):
            st.session_state.conversation_history = []
            st.session_state.chat_messages = []
            st.rerun()
    with col2:
        if st.button("💾 Export Chat", help="Download conversation history"):
            if st.session_state.chat_messages:
                chat_export = json.dumps(st.session_state.chat_messages, indent=2)
                st.download_button(
                    label="⬇️ Download",
                    data=chat_export,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No conversation to export yet")
    with col3:
        msg_count = len(st.session_state.chat_messages)
        st.metric("� Messages", msg_count)
    
    st.markdown("---")
    
    # Chat history display in a scrollable container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_messages:
            st.info("👋 **Start a conversation!** Ask me to generate code, explain concepts, or make modifications based on your documentation.")
        else:
            # Display all chat messages
            for msg in st.session_state.chat_messages:
                if msg['role'] == 'user':
                    with st.container():
                        st.markdown("##### 👤 You")
                        st.markdown(f"*{msg['timestamp']}*")
                        st.info(msg['content'])
                else:  # assistant
                    with st.container():
                        st.markdown("##### 🤖 Assistant")
                        st.markdown(f"*{msg['timestamp']}*")
                        
                        # Display code in expandable section for better readability
                        if 'code' in msg and msg['code']:
                            with st.expander("✨ **Generated Code** (click to expand/collapse)", expanded=True):
                                st.code(msg['code'], language="python", line_numbers=True)
                                
                                # Download button for this specific code
                                st.download_button(
                                    label="⬇️ Download this code",
                                    data=msg['code'],
                                    file_name=f"code_{msg['timestamp'].replace(':', '-').replace(' ', '_')}.py",
                                    mime="text/plain",
                                    key=f"download_{msg['timestamp']}"
                                )
                        
                        # Display sources if available
                        if 'sources' in msg and msg['sources']:
                            with st.expander(f"📚 **View Sources** ({len(msg['sources'])} references)", expanded=False):
                                for source in msg['sources']:
                                    relevance_pct = f"{source['relevance_score']:.0%}"
                                    st.markdown(f"**📄 {source['file']} - {source['location']}** ({relevance_pct} match)")
                                    
                                    # Relevance indicator
                                    relevance_val = source['relevance_score']
                                    if relevance_val >= 0.9:
                                        st.success(f"🎯 Excellent Match: {relevance_pct}")
                                    elif relevance_val >= 0.7:
                                        st.info(f"✅ Good Match: {relevance_pct}")
                                    else:
                                        st.warning(f"⚠️ Moderate Match: {relevance_pct}")
                                    
                                    with st.expander("View excerpt", expanded=False):
                                        st.code(source['excerpt'], language=None)
                                    
                                    st.markdown("---")
                        
                        st.markdown("")  # Spacing
    
    st.markdown("---")
    
    # Input section at the bottom (always visible)
    st.markdown("### 💭 Your Message")
    
    # Use form for better UX
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_area(
            "Ask a question or describe what you need...",
            placeholder="Examples:\n• Create a FastAPI endpoint for user authentication\n• Can you add error handling to the previous code?\n• How would I modify this to use async/await?\n• Explain how the JWT token validation works",
            height=120,
            help="Ask follow-up questions naturally - I'll remember our conversation context!"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_btn = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
        with col2:
            st.caption("💡 Tip: Ask follow-up questions! I maintain conversation context.")
    
    # Handle message submission
    if submit_btn and query:
        # Add user message to chat
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': query,
            'timestamp': timestamp
        })
        
        # Build conversation context from history
        conversation_context = ""
        for msg in st.session_state.conversation_history:
            conversation_context += f"User: {msg['query']}\nAssistant: {msg['response'][:500]}...\n\n"
        
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        # Stream the response
        try:
            import sseclient
            import json
            
            status_placeholder.info("🤖 Connecting to server...")
            
            # Make streaming request
            response = requests.post(
                f"{API_URL}/generate-stream",
                data={
                    "query": query,
                    "top_k": top_k,
                    "conversation_history": conversation_context
                },
                stream=True,
                headers={'Accept': 'text/event-stream'}
            )
            
            if response.ok:
                accumulated_code = ""
                sources = []
                
                # Parse SSE stream
                client = sseclient.SSEClient(response)
                
                for event in client.events():
                    try:
                        data = json.loads(event.data)
                        event_type = data.get('type')
                        
                        if event_type == 'status':
                            status_placeholder.info(f"🤖 {data.get('message')}")
                        
                        elif event_type == 'sources':
                            sources = data.get('sources', [])
                            # Display sources in real-time
                            with sources_placeholder.expander(f"📚 **Found {len(sources)} Sources**", expanded=False):
                                for source in sources:
                                    relevance_pct = f"{source['relevance_score']:.0%}"
                                    st.markdown(f"**📄 {source['file']} - {source['location']}** ({relevance_pct} match)")
                        
                        elif event_type == 'code':
                            # Accumulate code chunks and display in real-time
                            accumulated_code += data.get('content', '')
                            response_placeholder.code(accumulated_code, language="python", line_numbers=True)
                        
                        elif event_type == 'done':
                            status_placeholder.success("✅ " + data.get('message'))
                            
                            # Add final assistant response to chat
                            st.session_state.chat_messages.append({
                                'role': 'assistant',
                                'content': 'Generated code based on your documentation',
                                'code': accumulated_code,
                                'sources': sources,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Update conversation history for context
                            st.session_state.conversation_history.append({
                                'query': query,
                                'response': accumulated_code[:1000],  # Store first 1000 chars
                                'timestamp': timestamp
                            })
                            
                            # Clear placeholders and rerun
                            time.sleep(1)
                            status_placeholder.empty()
                            response_placeholder.empty()
                            sources_placeholder.empty()
                            st.rerun()
                        
                        elif event_type == 'error':
                            status_placeholder.error(f"❌ Error: {data.get('message')}")
                            break
                    
                    except json.JSONDecodeError:
                        continue
            else:
                st.error(f"❌ Request failed with status {response.status_code}")
                
        except ImportError:
            # Fallback to non-streaming if sseclient is not available
            st.warning("⚠️ Streaming not available. Install sseclient-py: `pip install sseclient-py`")
            st.info("Falling back to non-streaming mode...")
            
            with st.spinner('🤖 Thinking and generating code...'):
                try:
                    response = requests.post(
                        f"{API_URL}/generate", 
                        data={
                            "query": query, 
                            "top_k": top_k,
                            "conversation_history": conversation_context
                        }
                    )
                    
                    if response.ok:
                        result = response.json()
                        
                        if result.get("error") == "no_documents":
                            st.warning("⚠️ No documents found. Please upload reference documents in the Knowledge Base tab first.")
                        else:
                            # Add assistant response to chat
                            st.session_state.chat_messages.append({
                                'role': 'assistant',
                                'content': 'Generated code based on your documentation',
                                'code': result.get("code", ""),
                                'sources': result.get("sources", []),
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Update conversation history for context
                            st.session_state.conversation_history.append({
                                'query': query,
                                'response': result.get("code", "")[:1000],  # Store first 1000 chars
                                'timestamp': timestamp
                            })
                            
                            # Rerun to display new messages
                            st.rerun()
                    else:
                        st.error(f"❌ Request failed with status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# =============================================================================
# TAB 3: Analytics
# =============================================================================
with tab3:
    st.header("📊 System Analytics")
    st.markdown("Coming soon: Analytics dashboard with usage stats, token consumption, and performance metrics.")
    
    st.info("🚧 This section is under development. Future features will include:")
    st.markdown("""
    - 📈 Query history and patterns
    - 💰 Token usage and costs
    - ⏱️ Performance trends
    - 📚 Knowledge base statistics
    - 🎯 Most accessed sources
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Built with ❤️ using Streamlit, FastAPI, OpenAI, and Qdrant
</div>
""", unsafe_allow_html=True)
