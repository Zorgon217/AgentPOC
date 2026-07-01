import os
import json
import re
import streamlit as st
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant" 

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("Missing API keys. Ensure GROQ_API_KEY and TAVILY_API_KEY are in your .env file.")
    st.stop()

# --- CLIENTS ---
@st.cache_resource
def get_clients():
    return Groq(api_key=GROQ_API_KEY), TavilyClient(api_key=TAVILY_API_KEY)

groq_client, tavily_client = get_clients()

# --- SYSTEM PROMPT (The Orchestration Contract) ---
SYSTEM_PROMPT = """You are an expert data extraction engine. Analyze the user's query and extract the core market sector, the primary intent, and any specific attributes into a strict JSON object.

CRITICAL RULES:
1. Output ONLY valid JSON. No markdown formatting, no explanations.
2. The JSON must strictly follow this schema:
{
  "sector": "String (e.g., Retail, Real Estate, Healthcare, Automotive)",
  "intent": "String (The core action, e.g., 'buy boots', 'rent house')",
  "attributes": {
     "key": "value (Dynamically generate keys based on context, e.g., 'budget', 'location', 'specifications')"
  }
}
3. If a specific attribute is not mentioned, do not include it in the 'attributes' object.
"""

# --- DOMAIN REGISTRY (The Simulated DB Boundary) ---
DOMAIN_REGISTRY = {
    "Retail": ["takealot.com", "makro.co.za", "loot.co.za", "bobshop.co.za"],
    "Real Estate": ["property24.com", "privateproperty.co.za", "rawson.co.za", "pamgolding.co.za"],
    "Automotive": ["autotrader.co.za", "cars.co.za", "webuycars.co.za"],
    "Default": ["takealot.com", "gumtree.co.za", "junkmail.co.za"]
}

# --- TOOL REGISTRY: QUERY STRATEGIES ---
def build_realestate_query(attributes):
    location = attributes.get('location', '')
    budget = attributes.get('budget', '')
    intent = attributes.get('intent', 'property')
    return f"{intent} in {location} {budget}".strip()

def build_retail_query(attributes):
    product = attributes.get('product', 'item')
    specs = " ".join(attributes.get('specifications', []))
    return f"buy {product} {specs}".strip()

def build_default_query(attributes):
    intent = attributes.get('intent', '')
    extra = " ".join(str(v) for v in attributes.values() if v != intent)
    return f"{intent} {extra}".strip()

QUERY_REGISTRY = {
    "Real Estate": build_realestate_query,
    "Retail": build_retail_query
}

# --- UNIFIED EXECUTION ENGINE ---
def execute_market_search(sector, attributes):
    # 1. Determine query strategy
    query_builder = QUERY_REGISTRY.get(sector, build_default_query)
    search_query = query_builder(attributes)
    
    # 2. Determine domain restrictions
    allowed_domains = DOMAIN_REGISTRY.get(sector, DOMAIN_REGISTRY["Default"])
    
    # 3. Initial API call with domain restrictions (Simulated DB)
    response = tavily_client.search(
        search_query, 
        max_results=9, 
        include_raw_content=False,
        include_domains=allowed_domains
    )
    results = response.get('results', [])
    
    # 4. Track if fallback was triggered
    fallback_triggered = False
    
    # 5. Fallback mechanism: If restricted domains yield nothing, search the broader web
    if not results:
        print(f"[Orchestration] Zero results in restricted domains. Triggering broad web fallback for: {search_query}")
        fallback_triggered = True
        response = tavily_client.search(
            search_query, 
            max_results=10, 
            include_raw_content=False
        )
        results = response.get('results', [])
        
    return results, allowed_domains, fallback_triggered

# --- UI LAYOUT ---
st.set_page_config(page_title="Agent Intent Engine Demo", layout="wide")

# --- CUSTOM CSS INJECTION ---
st.markdown(
    """
    <style>
    /* Centered Title */
    .main-title { 
        text-align: center; 
        padding-top: 0 px; 
        padding-bottom: 30px; 
        color: #1E293B;
    }
    
    /* Highlighted Chat Input */
    .stChatInputContainer {
        border: 2px solid #6366F1;
        border-radius: 12px;
        padding: 8px 12px;
        background-color: #F8FAFC;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15);
        transition: box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .stChatInputContainer:focus-within {
        border-color: #4F46E5;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
    }
    
    /* Result Cards */
    .stMarkdown h3 a {
        color: #4F46E5;
        text-decoration: none;
    }
    .stMarkdown h3 a:hover {
        text-decoration: underline;
    }
    
    /* Divider Styling */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 16px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<h1 class="main-title">Agent Logic & Orchestration Demo for Aided backend PoC</h1>',
    unsafe_allow_html=True
)
left_spacer, chat_container, right_spacer = st.columns([1, 2, 1])

with chat_container:
    if prompt := st.chat_input("Describe what you are looking for..."):
        st.write(f"**User Query:** {prompt}")
        
        # 1. PERCEPTION: Extract Data via Model
        with st.spinner("Agent analyzing intent..."):
            try:
                completion = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                raw_response = completion.choices[0].message.content
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                json_string = json_match.group(0) if json_match else raw_response
                extracted_data = json.loads(json_string)
                
                st.success("JSON Data Extracted Successfully!")
                
                # 2. COLLAPSED JSON VIEW
                with st.expander("View Extracted JSON baseline Data", expanded=False):
                    st.json(extracted_data)
                
            except Exception as e:
                st.error(f"Extraction Error: {str(e)}")
                st.stop()

        # 3. ACTION: Route via Unified Execution Engine
        sector = extracted_data.get("sector", "Unknown")
        attributes = extracted_data.get("attributes", {})
        
        with st.spinner(f"[No curated data] Agent searching live web with Tavily API {sector} options..."):
            try:
                # Execute the unified search engine
                search_results, searched_domains, fallback_triggered = execute_market_search(sector, attributes)
                
                # 4. RENDER: Display results with domain transparency
                st.subheader(f"Live web data results ({sector})")
                
                # Display which domains were searched
                if fallback_triggered:
                    st.caption(f"🔍 Tavily API searched: Broad Web (no results found in restricted domains)")
                else:
                    domains_str = ", ".join(searched_domains)
                    st.caption(f"🔍  Sector restricted domains searched by Tavily API: {domains_str}")
                
                if not search_results:
                    st.warning("No live web options found for this query.")
                else:
                    for result in search_results:
                        st.markdown(f"### [{result.get('title', 'Untitled')}]({result.get('url', '#')})")
                        st.caption(result.get('url'))
                        st.write(result.get('content', 'No summary available.'))
                        st.divider()
                        
            except Exception as e:
                st.error(f"Action Execution Error (Tavily API): {str(e)}")