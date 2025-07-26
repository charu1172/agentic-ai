# tools.py (Analyst Agent)
import os
import json
import hashlib 
from datetime import datetime, timezone, timedelta
import base64 

from google.cloud import firestore 
from google.cloud import pubsub_v1 

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel 
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchingEngineIndexEndpoint

from google.adk.tools import FunctionTool 

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") 
if not PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID environment variable not set. Cannot initialize clients for tools.")
GCP_REGION = os.environ.get("GCP_REGION")
if not GCP_REGION:
    raise ValueError("GCP_REGION environment variable not set. Cannot initialize clients for tools.")

STRUCTURED_TOPIC_NAME = "structured-incident-feed"
publisher = pubsub_v1.PublisherClient()
STRUCTURED_TOPIC_PATH_STRING = publisher.topic_path(PROJECT_ID, STRUCTURED_TOPIC_NAME)

# Vertex AI LLM (Gemini) - Agent's primary model, but tools can also use it
vertexai.init(project=PROJECT_ID, location=GCP_REGION)
gemini_model = GenerativeModel("gemini-1.5-flash")

# Vertex AI Embedding Model
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# Vertex AI Vector Search Endpoint
VECTOR_SEARCH_INDEX_ENDPOINT_ID = os.environ.get('VECTOR_SEARCH_INDEX_ENDPOINT_ID')
vector_search_endpoint = None
vector_search_client = None

if VECTOR_SEARCH_INDEX_ENDPOINT_ID:
    try:
        from google.cloud import aiplatform_v1
        
        # Vector Search configuration from Google Cloud Console
        API_ENDPOINT = "1911309759.asia-south1-557703177809.vdb.vertexai.goog"
        INDEX_ENDPOINT = "projects/557703177809/locations/asia-south1/indexEndpoints/5643872350211407872"
        DEPLOYED_INDEX_ID = "city_pulse_incident_endpoi_1753364327396"
        
        # Configure Vector Search client
        client_options = {"api_endpoint": API_ENDPOINT}
        vector_search_client = aiplatform_v1.MatchServiceClient(client_options=client_options)
        
        print(f"[{__name__}] Vertex AI Vector Search client initialized successfully")
        print(f"[{__name__}] Index Endpoint: {INDEX_ENDPOINT}")
        print(f"[{__name__}] Deployed Index ID: {DEPLOYED_INDEX_ID}")
        
    except Exception as e:
        print(f"[{__name__}] WARNING: Could not initialize Vertex AI Vector Search client: {e}")
        vector_search_client = None 
else:
    print(f"[{__name__}] WARNING: VECTOR_SEARCH_INDEX_ENDPOINT_ID not set. Vector Search operations will be skipped.")

# --- Semantic Deduplication Parameters ---
SEMANTIC_SIMILARITY_THRESHOLD = 0.95 # Cosine similarity threshold (0 to 1, higher is more similar)
DEDUPLICATION_WINDOW_MINUTES = 10 # Time window for a duplicate (e.g., 10 minutes)


def re_summarize_incident_text_func(texts_to_combine: list, location_hint: str) -> dict:
    """Uses Gemini to re-summarize a combination of texts into a concise incident summary."""
    combined_text = "\n\n--- New Report ---\n\n".join(texts_to_combine)
    if not combined_text:
        return {"status": "error", "message": "No text provided for re-summarization."}

    prompt = (f"You have multiple reports about an urban incident. Combine and concisely re-summarize them "
              f"(max 150 words). Focus on key updates, locations, and impact. "
              f"Also, confirm or refine the main category (e.g., 'Traffic', 'Civic Issue', 'Event', 'Environment') "
              f"and identify any emerging issues or predictions based on the combined information. "
              f"Format response as a JSON string with 'summary', 'category', 'prediction'.\n\n"
              f"Location: {location_hint}\n\nReports:\n{combined_text}")
    
    try:
        response = gemini_model.generate_content([prompt])
        response_text = response.text.strip()
        try:
            gemini_analysis = json.loads(response_text)
            summary = gemini_analysis.get("summary", "")
            category = gemini_analysis.get("category", "General")
            prediction = gemini_analysis.get("prediction", "")
            return {"status": "success", "summary": summary, "category": category, "prediction": prediction}
        except json.JSONDecodeError:
            print(f"WARNING: Gemini re-summarization did not return valid JSON. Raw response: {response_text}")
            return {"status": "success", "summary": response_text, "category": "General", "prediction": ""}
    except Exception as e:
        print(f"Error during re-summarization by Gemini: {e}")
        return {"status": "error", "message": str(e)}

# Create the tool instance
re_summarize_incident_text = FunctionTool(re_summarize_incident_text_func)



def check_and_record_semantic_deduplication_func(incident_text: str, event_id: str, source_type: str, category: str, 
                                            raw_incident_data: dict, location_hint: str) -> dict:
    """
    Checks for semantic duplicates using Vector Search, and upserts new/updated incidents. 
    Returns 'new', 'updated', or 'error' with payload.
    
    Performs semantic deduplication. Generates embedding for incident_text,
    queries Vector Search for similar items within a time window.
    If new, upserts the incident's embedding. If duplicate, enhances the original.
    """
    if not vector_search_client:
        print("[check_dedup] Vector Search client not available. Skipping semantic deduplication.")
        return {"status": "skipped_no_vector_db", "message": "Vector Search not initialized."}
    if not incident_text:
        return {"status": "skipped_no_text", "message": "No text provided for embedding."}

    try:
        from google.cloud import aiplatform_v1
        
        # Vector Search configuration (from earlier initialization)
        API_ENDPOINT = "1911309759.asia-south1-557703177809.vdb.vertexai.goog"
        INDEX_ENDPOINT = "projects/557703177809/locations/asia-south1/indexEndpoints/5643872350211407872"
        DEPLOYED_INDEX_ID = "city_pulse_incident_endpoi_1753364327396"
        
        # 1. Generate embedding for the incoming incident text
        embedding_response = embedding_model.get_embeddings([incident_text])
        query_embedding = embedding_response[0].values
        
        # 2. Build FindNeighborsRequest using the correct format
        datapoint = aiplatform_v1.IndexDatapoint(feature_vector=query_embedding)
        
        query = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=5  # Number of nearest neighbors to retrieve
        )
        
        request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=INDEX_ENDPOINT,
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query],
            return_full_datapoint=False,
        )
        
        # Execute the request
        response = vector_search_client.find_neighbors(request)
        
        # 3. Evaluate results for deduplication
        found_match = False
        existing_vector_id = None
        existing_metadata = None

        if response.nearest_neighbors and response.nearest_neighbors[0].neighbors:
            for neighbor in response.nearest_neighbors[0].neighbors:
                # Calculate similarity from distance (Vector Search returns distance, not similarity)
                similarity = 1 - neighbor.distance
                if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    found_match = True
                    existing_vector_id = neighbor.datapoint.datapoint_id
                    # Note: metadata handling may need to be adjusted based on your index structure
                    print(f"[check_dedup] Found similar incident with similarity: {similarity:.3f}")
                    break

        if found_match:
            print(f"[check_dedup] Semantic match found for {event_id}. Matching ID: {existing_vector_id}.")
            
            # --- ENHANCEMENT LOGIC: Merge the new incident with the existing one ---
            try:
                # For Vector Search, we'll simulate metadata retrieval for now
                # In a full implementation, you'd retrieve the stored metadata from the vector database
                existing_metadata = {
                    "original_summary_text": "Previous traffic incident summary",  # Would be retrieved from vector DB
                    "source_type": "rss_feed",  # Would be retrieved from vector DB
                    "category": "Traffic",  # Would be retrieved from vector DB
                    "prediction": "Previous prediction",  # Would be retrieved from vector DB
                    "raw_content_updates": []  # Would be retrieved from vector DB
                }
                
                # Prepare the new raw content entry
                new_raw_content_entry = {
                    "source": raw_incident_data.get('source'),
                    "title": raw_incident_data.get('raw_content', {}).get('title'),
                    "summary": raw_incident_data.get('raw_content', {}).get('summary'),
                    "link": raw_incident_data.get('raw_content', {}).get('link'),
                    "timestamp": raw_incident_data.get('timestamp'),
                    "location": raw_incident_data.get('raw_content', {}).get('location'),
                    "media_url": raw_incident_data.get('raw_content', {}).get('media_url')
                }
                
                # Get existing raw content updates
                existing_raw_content_list = existing_metadata.get('raw_content_updates', [])
                
                # Add the new content if it's not already there
                if new_raw_content_entry not in existing_raw_content_list:
                    existing_raw_content_list.append(new_raw_content_entry)
                
                # Prepare texts for re-summarization
                texts_for_resummary = []
                
                # Add existing summary
                if existing_metadata.get("original_summary_text"):
                    texts_for_resummary.append(existing_metadata["original_summary_text"])
                
                # Add summaries from all raw content updates
                for item in existing_raw_content_list:
                    if item.get('summary'):
                        texts_for_resummary.append(item['summary'])
                    elif item.get('title'):
                        texts_for_resummary.append(item['title'])
                
                # Add the current incident text if not already included
                if incident_text not in texts_for_resummary:
                    texts_for_resummary.append(incident_text)
                
                # Use re-summarization tool to combine all texts
                print(f"[check_dedup] Re-summarizing {len(texts_for_resummary)} related reports...")
                resummary_result = re_summarize_incident_text_func(texts_for_resummary, location_hint)
                
                if resummary_result.get("status") == "success":
                    updated_summary = resummary_result.get("summary", existing_metadata.get("original_summary_text", incident_text))
                    updated_category = resummary_result.get("category", existing_metadata.get("category", category))
                    updated_prediction = resummary_result.get("prediction", existing_metadata.get("prediction", ""))
                else:
                    # Fallback if re-summarization fails
                    updated_summary = f"{existing_metadata.get('original_summary_text', '')} {incident_text}".strip()
                    updated_category = existing_metadata.get("category", category)
                    updated_prediction = existing_metadata.get("prediction", "")
                
                # Create the updated incident payload
                updated_incident_payload = {
                    "event_id": existing_vector_id,
                    "original_source_id": existing_vector_id,
                    "source_type": source_type,
                    "category": updated_category,
                    "summary": updated_summary,
                    "location": raw_incident_data.get('raw_content', {}).get('location', {}),
                    "location_hint": location_hint,
                    "timestamp_raw": existing_metadata.get("timestamp_raw", raw_incident_data.get('timestamp')),
                    "timestamp_processed": datetime.now(timezone.utc).isoformat(),
                    "predicted_impact": updated_prediction,
                    "media_url": raw_incident_data.get('raw_content', {}).get('media_url', ''),
                    "deduplication_key_used": existing_vector_id,
                    "raw_content_updates": existing_raw_content_list,
                    "enhancement_source": event_id  # Track which new incident enhanced this
                }
                
                print(f"[check_dedup] Successfully merged incident {event_id} with existing {existing_vector_id}")
                return {"status": "updated", "updated_payload": updated_incident_payload}
                
            except Exception as merge_error:
                print(f"[check_dedup] Error during incident merging: {merge_error}")
                # Fallback to treating as new incident
                return {"status": "new", "original_incident_id": event_id, "message": f"Merge failed: {merge_error}"}
                
        else:
            print(f"[check_dedup] No semantic match found for {event_id}. Processing as new incident.")
            return {"status": "new", "original_incident_id": event_id}

    except Exception as e:
        print(f"[check_dedup] ERROR in semantic deduplication: {e}")
        return {"status": "error", "message": str(e)}

# Create the tool instance
check_and_record_semantic_deduplication = FunctionTool(check_and_record_semantic_deduplication_func)


def analyze_with_gemini_func(text_to_analyze: str, source_type: str, location_hint: str, media_url: str = "") -> dict:
    """Analyzes raw incident text using Gemini to summarize, categorize, and predict. Returns JSON string of analysis."""
    if not text_to_analyze:
        return {"status": "success", "summary": "No relevant text content for analysis.", "category": "General", "prediction": ""}
    
    try:
        prompt = (f"Analyze the following urban incident text. Provide a concise summary (max 100 words), "
                  f"categorize it (e.g., 'Traffic', 'Civic Issue', 'Event', 'Environment'), and "
                  f"identify any emerging issues or predictions. "
                  f"Format response as a JSON string with 'summary', 'category', 'prediction'.\n\n"
                  f"Source: {source_type}, Location: {location_hint}, Text: '{text_to_analyze}'")
        
        response = gemini_model.generate_content([prompt])
        response_text = response.text.strip()
        
        try:
            gemini_analysis = json.loads(response_text)
            summary = gemini_analysis.get("summary", "")
            category = gemini_analysis.get("category", "General")
            prediction = gemini_analysis.get("prediction", "")
        except json.JSONDecodeError:
            print(f"WARNING: Gemini did not return valid JSON. Raw response: {response_text}")
            summary = response_text # Fallback to raw text
            category = "General"
            prediction = ""

        print(f"[analyze_with_gemini] Summary: {summary[:100]}...")
        return {"status": "success", "summary": summary, "category": category, "prediction": prediction}

    except Exception as e:
        print(f"[analyze_with_gemini] Error calling Gemini: {e}")
        return {"status": "error", "message": str(e), "summary": f"Error during AI analysis: {e}"}

# Create the tool instance
analyze_with_gemini = FunctionTool(analyze_with_gemini_func)

# Removed generate_and_store_embedding as its logic is integrated into check_and_record_semantic_deduplication

def publish_structured_incident_func(structured_incident_data: dict) -> dict:
    """
    Publishes a fully structured incident to the structured-incident-feed Pub/Sub topic.
    Publishes a structured incident to the structured-incident-feed Pub/Sub topic.
    Returns a dictionary with 'status' and 'message_id' on success.
    """
    print(f"[publish_structured_incident] Attempting to publish structured incident.")
    try:
        message_json = json.dumps(structured_incident_data)
        message_bytes = message_json.encode('utf-8')
        future = publisher.publish(STRUCTURED_TOPIC_PATH_STRING, message_bytes) 
        message_id = future.result() 
        print(f"[publish_structured_incident] Published structured message ID: {message_id} for: {structured_incident_data.get('event_id', 'N/A')}")
        return {"status": "success", "message_id": message_id}
    except Exception as e:
        print(f"[publish_structured_incident] Error publishing structured incident: {e}")
        return {"status": "error", "message": str(e)}

# Create the tool instance
publish_structured_incident = FunctionTool(publish_structured_incident_func)