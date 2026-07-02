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
# FIX 1: Removed trailing spaces from the JSON schema example to prevent the LLM from hallucinating spaced keys.
SYSTEM_PROMPT = """You are an expert marketplace intent and data extraction engine.
Analyze the user's query and extract the market sector, primary intent, provided information, and missing information.
CRITICAL RULES:
Output ONLY valid JSON. No markdown formatting, no explanations.
Use this exact top-level schema:
{
 "sector": "String (e.g., Retail, Real Estate, Healthcare, Automotive)",
 "intent": "String (short normalized action, e.g., 'buy_product', 'rent_property', 'buy_used_vehicle')",
 "provided_information": {
 "key": "value (Dynamically generate keys based on context)"
},
 "missing_information": {
 "critical": ["Only fields absolutely required for a useful/valid search"],
 "useful": ["Fields that would improve the search but should not block it"],
 "optional": ["Nice-to-have preferences"]
},
 "followup_question": "A single natural question asking only for critical missing information, or null if no critical information is missing"
}
If a specific attribute is not mentioned, do not include it in provided_information.
Do not ask for optional preferences unless the search would be too broad, misleading, or unusable without them.
For South African queries, preserve currency as ZAR/Rands/R where mentioned.
"""

# --- DOMAIN REGISTRY (The Simulated DB Boundary) ---
# FIX 1: Stripped all trailing/leading whitespace from keys and values.
DOMAIN_REGISTRY = {
    "Retail": ["takealot.com", "makro.co.za", "loot.co.za", "bobshop.co.za"],
    "Real Estate": ["property24.com", "privateproperty.co.za", "rawson.co.za", "pamgolding.co.za"],
    "Automotive": ["autotrader.co.za", "cars.co.za", "webuycars.co.za"],
    "Default": ["takealot.com", "gumtree.co.za", "junkmail.co.za"]
}

# --- FIELD NORMALIZATION ---
# FIX 1: Stripped all trailing/leading whitespace from keys and alias lists.
FIELD_ALIASES = {
    "location": [
        "location", "city", "town", "suburb", "area", "province", "where",
        "place", "region"
    ],
    "budget": [
        "budget", "price", "price_range", "max_price", "min_price", "amount",
        "rent", "monthly_rent", "budget_range", "maximum_budget", "minimum_budget"
    ],
    "product": [
        "product", "product_type", "item", "item_type", "keyword", "search_term",
        "query", "goods", "article", "thing", "category"
    ],
    "size": [
        "size", "shoe_size", "clothing_size", "footwear_size", "boot_size"
    ],
    "vehicle_type": [
        "vehicle_type", "car_type", "body_type", "vehicle", "car", "bakkie",
        "automotive_type", "vehicle_category"
    ],
    "property_type": [
        "property_type", "dwelling_type", "home_type", "flat", "apartment",
        "house", "outbuilding", "unit", "room", "accommodation_type"
    ],
    "bedrooms": [
        "bedrooms", "beds", "rooms", "number_of_bedrooms", "bedroom_count"
    ],
}

def normalize_field_name(key):
    """Normalize raw LLM keys so aliases match despite spaces, hyphens, or casing."""
    return re.sub(r"[^a-z0-9]+", "", str(key).strip().lower()).strip()

# --- VALIDATION RULES ---
# FIX 1: Stripped all trailing/leading whitespace.
VALIDATION_RULES = {
    "Automotive": {
        "buy_used_vehicle": {
            "critical": ["vehicle_type", "location", "budget"],
            "useful": ["make", "model", "year", "mileage", "transmission", "fuel_type"]
        },
        "buy_vehicle": {
            "critical": ["vehicle_type", "location", "budget"],
            "useful": ["condition", "make", "model", "year", "mileage", "transmission"]
        }
    },
    "Real Estate": {
        "rent_property": {
            "critical": ["location", "budget"],
            "useful": ["property_type", "bedrooms", "parking", "utilities_included"]
        },
        "buy_property": {
            "critical": ["location", "budget"],
            "useful": ["property_type", "bedrooms", "parking"]
        }
    },
    "Retail": {
        "buy_product": {
            "critical": ["product"],
            "useful": ["budget", "brand", "size", "colour", "material"]
        }
    }
}

def normalize_sector(sector):
    # FIX 1: Returns clean, trimmed strings without trailing spaces.
    if not sector:
        return "Default"
    sector_clean = str(sector).strip().lower()
    if "real" in sector_clean or "property" in sector_clean or "estate" in sector_clean:
        return "Real Estate"
    if "auto" in sector_clean or "vehicle" in sector_clean or "car" in sector_clean:
        return "Automotive"
    if "retail" in sector_clean or "shopping" in sector_clean or "product" in sector_clean:
        return "Retail"
    return str(sector).strip().title()

def normalize_intent(intent, sector):
    intent_clean = str(intent or "").strip().lower().replace(" ", "").replace("-", "")
    if sector == "Automotive" and "buy" in intent_clean:
        if "used" in intent_clean or "second" in intent_clean or "preowned" in intent_clean:
            return "buy_used_vehicle"
        return "buy_vehicle"
    if sector == "Real Estate":
        if "rent" in intent_clean or "lease" in intent_clean:
            return "rent_property"
        if "buy" in intent_clean or "purchase" in intent_clean:
            return "buy_property"
    if sector == "Retail" and ("buy" in intent_clean or "purchase" in intent_clean):
        return "buy_product"
    return intent_clean or "unknown"

def canonicalize_attributes(attributes):
    attributes = attributes or {}
    canonical = dict(attributes)
    normalized_key_map = {normalize_field_name(k): k for k in attributes.keys()}
    for canonical_key, aliases in FIELD_ALIASES.items():
        existing_value = canonical.get(canonical_key)
        if existing_value not in (None, "", [], {}):
            continue
        for alias in aliases:
            source_key = normalized_key_map.get(normalize_field_name(alias))
            if source_key is None:
                continue
            source_value = attributes.get(source_key)
            if source_value not in (None, "", [], {}):
                canonical[canonical_key] = source_value
                break
    return canonical
#=================================================================

def has_field(attributes, field_name):
    """Check if a field has a meaningful value (not None, empty, or placeholder strings)."""
    value = attributes.get(field_name)
    if value in (None, "", [], {}):
        return False
    # FIX: Treat common LLM placeholder strings as missing data
    if isinstance(value, str) and value.strip().lower() in ("not specified", "n/a", "unknown", "null", "none"):
        return False
    return True

def infer_missing_critical(sector, intent, attributes):
    sector_rules = VALIDATION_RULES.get(sector, {})
    intent_rules = sector_rules.get(intent)
    if not intent_rules:
        return [], []
    missing_critical = [field for field in intent_rules.get("critical", []) if not has_field(attributes, field)]
    useful = [field for field in intent_rules.get("useful", []) if not has_field(attributes, field)]
    return missing_critical, useful

#==================================================================

def build_followup_question(sector, intent, missing_critical):
    if not missing_critical:
        return None
    labels = {
        "budget": "your budget range",
        "location": "the location or area",
        "vehicle_type": "what type of vehicle you want",
        "property_type": "what type of property you want",
        "bedrooms": "how many bedrooms you need",
        "product": "what product you are looking for",
        "size": "the size you need",
    }
    missing_labels = [labels.get(field, field.replace("_", " ")) for field in missing_critical]
    if len(missing_labels) == 1:
        return f"To get useful {sector.lower()} results, please tell me {missing_labels[0]}."
    if len(missing_labels) == 2:
        joined = " and ".join(missing_labels)
    else:
        joined = ", ".join(missing_labels[:-1]) + f", and {missing_labels[-1]}"
    return f"To get useful {sector.lower()} results, please tell me {joined}."

def normalize_extracted_data(extracted_data):
    # FIX 2: Enforced strict .get() fallbacks with safe defaults to prevent KeyError crashes on LLM schema hallucinations.
    sector = normalize_sector(extracted_data.get("sector", "Default"))
    intent = normalize_intent(extracted_data.get("intent", "unknown"), sector)
    
    attributes = extracted_data.get("provided_information") or extracted_data.get("attributes") or {}
    attributes = canonicalize_attributes(attributes)
    
    missing_critical, missing_useful = infer_missing_critical(sector, intent, attributes)
    can_search_now = len(missing_critical) == 0
    followup_question = None if can_search_now else build_followup_question(sector, intent, missing_critical)
    
    normalized = {
        "sector": sector,
        "intent": intent,
        "provided_information": attributes,
        "missing_information": {
            "critical": missing_critical,
            "useful": missing_useful,
            "optional": []
        },
        "can_search_now": can_search_now,
        "followup_question": followup_question
    }
    return normalized

# --- TOOL REGISTRY: QUERY STRATEGIES ---
def safe_join(values):
    """Join values while filtering out placeholder strings."""
    if isinstance(values, list):
        # FIX: Filter out placeholder strings from lists
        filtered = [str(v) for v in values if v and str(v).strip().lower() not in ("not specified", "n/a", "unknown", "null", "none")]
        return " ".join(filtered)
    # FIX: Return empty string for placeholder values
    if not values or str(values).strip().lower() in ("not specified", "n/a", "unknown", "null", "none"):
        return ""
    return str(values)

def build_realestate_query(attributes, intent):
    location = safe_join(attributes.get("location"))
    budget = safe_join(attributes.get("budget"))
    property_type = safe_join(attributes.get("property_type"))
    bedrooms = safe_join(attributes.get("bedrooms"))
    action = "rent" if intent == "rent_property" else "buy"
    bedroom_text = f"{bedrooms} bedroom" if bedrooms else ""
    return f"{action} {bedroom_text} {property_type} in {location} {budget}".strip()

def build_retail_query(attributes, intent):
    product = safe_join(attributes.get("product", "item"))
    size = safe_join(attributes.get("size"))
    material = safe_join(attributes.get("material"))
    brand = safe_join(attributes.get("brand"))
    budget = safe_join(attributes.get("budget"))
    return f"buy {brand} {material} {product} size {size} {budget}".strip()

def build_automotive_query(attributes, intent):
    location = safe_join(attributes.get("location"))
    budget = safe_join(attributes.get("budget"))
    vehicle_type = safe_join(attributes.get("vehicle_type", "car"))
    make = safe_join(attributes.get("make"))
    model = safe_join(attributes.get("model"))
    condition = "used second hand" if intent == "buy_used_vehicle" else safe_join(attributes.get("condition"))
    return f"buy {condition} {make} {model} {vehicle_type} in {location} {budget}".strip()

def build_default_query(attributes, intent):
    extra = " ".join(str(v) for v in attributes.values() if v)
    return f"{intent.replace('_', ' ')} {extra}".strip()

QUERY_REGISTRY = {
    "Real Estate": build_realestate_query,
    "Retail": build_retail_query,
    "Automotive": build_automotive_query,
}

# --- UNIFIED EXECUTION ENGINE ---
def execute_market_search(sector, intent, attributes):
    query_builder = QUERY_REGISTRY.get(sector, build_default_query)
    search_query = query_builder(attributes, intent)
    allowed_domains = DOMAIN_REGISTRY.get(sector, DOMAIN_REGISTRY["Default"])
    
    response = tavily_client.search(
        search_query,
        max_results=9,
        include_raw_content=False,
        include_domains=allowed_domains
    )
    results = response.get("results", [])
    
    fallback_triggered = False
    if not results:
        print(f"[Orchestration] Zero results in restricted domains. Triggering broad web fallback for: {search_query}")
        fallback_triggered = True
        response = tavily_client.search(
            search_query,
            max_results=10,
            include_raw_content=False
        )
        results = response.get("results", [])
        
    return results, allowed_domains, fallback_triggered, search_query

# --- UI LAYOUT ---
st.set_page_config(page_title="Agent Intent Engine Demo", layout="wide")

# --- SESSION STATE & LOOP SAFEGUARDS ---
if "pending_request" not in st.session_state:
    st.session_state.pending_request = None
if "original_query" not in st.session_state:
    st.session_state.original_query = ""
# FIX 3: Added attempt_count to prevent infinite clarification loops.
if "attempt_count" not in st.session_state:
    st.session_state.attempt_count = 0

# --- CUSTOM CSS INJECTION ---
st.markdown(
    """
    <style>
    .main-title { text-align: center; padding-top: 0px; padding-bottom: 30px; color: #1E293B; }
    .stChatInputContainer { border: 2px solid #6366F1; border-radius: 12px; padding: 8px 12px; background-color: #F8FAFC; box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15); transition: box-shadow 0.3s ease, border-color 0.3s ease; }
    .stChatInputContainer:focus-within { border-color: #4F46E5; box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3); }
    .stMarkdown h3 a { color: #4F46E5; text-decoration: none; }
    .stMarkdown h3 a:hover { text-decoration: underline; }
    hr { border: none; border-top: 1px solid #E2E8F0; margin: 16px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">Agent Logic & Orchestration Demo for Aided backend PoC</h1>', unsafe_allow_html=True)

left_spacer, chat_container, right_spacer = st.columns([1, 2, 1])

with chat_container:
    if st.session_state.pending_request:
        st.info(st.session_state.pending_request.get("followup_question"))
        st.caption(f"Attempt {st.session_state.attempt_count}/2 to provide critical information.")

    if prompt := st.chat_input("Describe what you are looking for..."):
        pending_request = st.session_state.pending_request
        
        # FIX 3: Context persistence and loop safeguard logic.
        if pending_request:
            st.session_state.attempt_count += 1
            st.write(f"**Additional Information (Attempt {st.session_state.attempt_count}):** {prompt}")
            
            # Force broad search if user fails to clarify after 2 attempts
            if st.session_state.attempt_count >= 2:
                st.warning("Maximum clarification attempts reached. Executing broad search with available data.")
                combined_prompt = st.session_state.original_query
                st.session_state.pending_request = None
                st.session_state.attempt_count = 0
            else:
                combined_prompt = f"{st.session_state.original_query}\nAdditional user information: {prompt}"
        else:
            st.write(f"**User Query:** {prompt}")
            st.session_state.original_query = prompt
            st.session_state.attempt_count = 0
            combined_prompt = prompt

        # 1. PERCEPTION: Extract Data via Model
        with st.spinner("Agent analyzing intent and required information..."):
            try:
                completion = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": combined_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw_response = completion.choices[0].message.content
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                json_string = json_match.group(0) if json_match else raw_response
                extracted_data = json.loads(json_string)
                extracted_data = normalize_extracted_data(extracted_data)
                st.success("JSON Data Extracted Successfully!")
                
                with st.expander("View Extracted JSON baseline Data", expanded=False):
                    st.json(extracted_data)
            except Exception as e:
                st.error(f"Extraction Error: {str(e)}")
                st.stop()

        # 3. VALIDATION GATE: Ask only if critical information is missing
        if not extracted_data.get("can_search_now", True):
            st.session_state.pending_request = {
                "original_query": st.session_state.original_query,
                "extracted_data": extracted_data,
                "followup_question": extracted_data.get("followup_question")
            }
            st.warning(extracted_data.get("followup_question"))
            st.caption("Search paused because essential information is missing. Optional preferences will not block the search.")
            st.stop()

        # 4. ACTION: Route via Unified Execution Engine
        sector = extracted_data.get("sector", "Default")
        intent = extracted_data.get("intent", "unknown")
        attributes = extracted_data.get("provided_information", {})
        
        with st.spinner(f"Agent searching live web with Tavily API {sector} options..."):
            try:
                search_results, searched_domains, fallback_triggered, search_query = execute_market_search(sector, intent, attributes)
                
                st.subheader(f"Live web data results ({sector})")
                st.caption(f"Generated search query: {search_query}")
                
                if fallback_triggered:
                    st.caption("🔍 Tavily API searched: Broad Web (no results found in restricted domains)")
                else:
                    domains_str = ", ".join(searched_domains)
                    st.caption(f" Sector restricted domains searched by Tavily API: {domains_str}")
                    
                useful_missing = extracted_data.get("missing_information", {}).get("useful", [])
                if useful_missing:
                    st.caption(f"You can narrow results later with: {', '.join(useful_missing)}")
                    
                if not search_results:
                    st.warning("No live web options found for this query.")
                else:
                    for result in search_results:
                        st.markdown(f"### [{result.get('title', 'Untitled')}]({result.get('url', '#')})")
                        st.caption(result.get("url"))
                        st.write(result.get("content", "No summary available."))
                        st.divider()
                        
                # Clear state on successful completion
                st.session_state.pending_request = None
                st.session_state.attempt_count = 0
                
            except Exception as e:
                st.error(f"Action Execution Error (Tavily API): {str(e)}")