import os
import json
import hashlib 
import numpy as np
from datetime import datetime, timezone, timedelta
import base64 

from google.cloud import firestore 
from google.cloud import pubsub_v1 

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel 
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchingEngineIndexEndpoint

from google.cloud import aiplatform_v1 

from google.adk.tools import FunctionTool 


# --- Global Configurations for Tools ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID") 
if not PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID environment variable not set. Cannot initialize clients for tools.")
GCP_REGION = os.environ.get("GCP_REGION")
if not GCP_REGION:
    raise ValueError("GCP_REGION environment variable not set. Cannot initialize clients for tools.")

# Pub/Sub Client for Structured Incident Feed
STRUCTURED_TOPIC_NAME = "structured-incident-feed"
publisher = pubsub_v1.PublisherClient()
STRUCTURED_TOPIC_PATH_STRING = publisher.topic_path(PROJECT_ID, STRUCTURED_TOPIC_NAME)

# Vertex AI LLM (Gemini) - Agent's primary model, but tools can also use it
vertexai.init(project=PROJECT_ID, location=GCP_REGION)
gemini_model = GenerativeModel("gemini-1.5-flash") # Match agent.py model

# Vertex AI Embedding Model
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# Vertex AI Vector Search Client & Endpoint Initialization
# Use environment variables for proper deployment configuration
# Renamed variables for clarity
VECTOR_SEARCH_API_ENDPOINT = os.environ.get('VECTOR_SEARCH_API_ENDPOINT') # e.g., "1645951261.asia-south1-557703177809.vdb.vertexai.goog"
VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME = os.environ.get('VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME') # e.g., "projects/557703177809/locations/asia-south1/indexEndpoints/5643872350211407872"
VECTOR_SEARCH_DEPLOYED_INDEX_ID = os.environ.get('VECTOR_SEARCH_DEPLOYED_INDEX_ID') # e.g., "9176946257883561984"

vector_search_client = None
if VECTOR_SEARCH_API_ENDPOINT and VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME and VECTOR_SEARCH_DEPLOYED_INDEX_ID:
    try:
        client_options = {"api_endpoint": VECTOR_SEARCH_API_ENDPOINT}
        vector_search_client = aiplatform_v1.MatchServiceClient(client_options=client_options)
        
        print(f"[{__name__}] Vertex AI Vector Search client initialized successfully from env vars.")
        print(f"[{__name__}] API Endpoint: {VECTOR_SEARCH_API_ENDPOINT}")
        print(f"[{__name__}] Resource Name: {VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME}")
        print(f"[{__name__}] Deployed Index ID: {VECTOR_SEARCH_DEPLOYED_INDEX_ID}")
        
    except Exception as e:
        print(f"[{__name__}] WARNING: Could not initialize Vector Search client from env vars: {e}")
        vector_search_client = None 
else:
    print(f"[{__name__}] WARNING: Vector Search env vars (API_ENDPOINT, RESOURCE_NAME, DEPLOYED_INDEX_ID) not fully set. Vector Search operations will be skipped.")

# --- Semantic Deduplication Parameters ---
SEMANTIC_SIMILARITY_THRESHOLD = 0.95 # Cosine similarity threshold (0 to 1, higher is more similar)
DEDUPLICATION_WINDOW_MINUTES = 10 # Time window for a duplicate (e.g., 10 minutes)


# --- Tool Definition: re_summarize_incident_text ---
def re_summarize_incident_text_func(texts_to_combine: list, location_hint: str) -> dict:
    """Combines multiple text snippets and asks Gemini to create a single, updated summary."""
    combined_text = "\n\n--- New Report ---\n\n".join(texts_to_combine)
    if not combined_text:
        return {"status": "error", "message": "No text provided for re-summarization."}

    prompt = (f"You have multiple reports about an urban incident. Combine and concisely re-summarize them "
              f"(max 150 words). Focus on key updates, locations, and impact. "
              f"Also, confirm or refine the main category (e.g., 'Traffic', 'Civic Issue', 'Event', 'Environment'), " # Corrected comma
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

re_summarize_incident_text = FunctionTool(re_summarize_incident_text_func)



# --- Tool Definition: check_and_record_semantic_deduplication ---
def check_and_record_semantic_deduplication_func(incident_text: str, event_id: str, source_type: str, category: str, 
                                            raw_incident_data: dict, location_hint: str) -> dict: 
    """
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
        # 1. Generate embedding for the incoming incident text
        embedding_response = embedding_model.get_embeddings([incident_text])
        raw_embedding = np.array(embedding_response[0].values)  # Get embedding as numpy array

        # --- CRITICAL CORRECTION: Perform L2 Normalization ---
        norm = np.linalg.norm(raw_embedding)
        if norm == 0:  # Handle zero vector case to avoid division by zero
            query_embedding = raw_embedding.tolist()  # Keep as is for zero vector
        else:
            query_embedding = (raw_embedding / norm).tolist()  # L2 normalize
        
        # 2. Build FindNeighborsRequest with proper filtering and metadata retrieval
        now_utc_timestamp_int = int(datetime.now(timezone.utc).timestamp()) 
        time_threshold_timestamp_int = int((datetime.now(timezone.utc) - timedelta(minutes=DEDUPLICATION_WINDOW_MINUTES)).timestamp())

        # Construct FindNeighborsRequest.Query
        query_datapoint = aiplatform_v1.IndexDatapoint(feature_vector=query_embedding)
        
        query_obj = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=query_datapoint,
            neighbor_count=5,  # Number of nearest neighbors to retrieve
            numeric_restricts=[ 
                aiplatform_v1.IndexDatapoint.NumericRestriction(
                    namespace="timestamp_processed",
                    value_int=time_threshold_timestamp_int, # Use value_int for timestamp consistency
                    op=aiplatform_v1.IndexDatapoint.NumericRestriction.Operator.GREATER_EQUAL
                )
            ],
            restricts=[ 
                aiplatform_v1.IndexDatapoint.Restriction(
                    namespace="source_type",
                    allow_list=[source_type]
                ),
                aiplatform_v1.IndexDatapoint.Restriction(
                    namespace="category",
                    allow_list=[category]
                )
            ]
        )
        
        request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME, # Use env var resource name
            deployed_index_id=VECTOR_SEARCH_DEPLOYED_INDEX_ID, # Use env var deployed index ID
            queries=[query_obj],
            return_full_datapoint=True,  # Critical: needed to get metadata_json
        )
        
        # Execute the request
        response = vector_search_client.find_neighbors(request)
        
        # 3. Evaluate results for deduplication
        found_match = False
        existing_vector_id = None
        existing_metadata = None

        if response.nearest_neighbors and response.nearest_neighbors[0].neighbors:
            for neighbor in response.nearest_neighbors[0].neighbors:
                similarity = 1 - neighbor.distance
                if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    found_match = True
                    existing_vector_id = neighbor.datapoint.datapoint_id
                    try:
                        existing_metadata = json.loads(neighbor.datapoint.metadata_json)
                        print(f"[check_dedup] Found similar incident with similarity: {similarity:.3f}")
                        print(f"[check_dedup] Retrieved metadata for existing incident: {existing_vector_id}")
                        break
                    except (json.JSONDecodeError, AttributeError) as meta_error:
                        print(f"[check_dedup] Error parsing metadata for {existing_vector_id}: {meta_error}. Skipping neighbor.")
                        continue # Continue to next neighbor if metadata is corrupted

        if found_match:
            print(f"[check_dedup] Semantic match found for {event_id}. Matching ID: {existing_vector_id}.")
            
            try:
                if not existing_metadata:
                    print(f"[check_dedup] ERROR: No valid metadata found for existing incident {existing_vector_id}. Cannot merge.")
                    return {"status": "error", "message": f"Merge failed: No metadata for {existing_vector_id}"}
                
                print(f"[check_dedup] Processing enhancement for existing incident: {existing_vector_id}")
                
                existing_raw_content_list = existing_metadata.get('raw_content_updates', []) 
                
                new_raw_content_entry = {
                    "source": raw_incident_data.get('source'),
                    "title": raw_incident_data.get('raw_content', {}).get('title'),
                    "summary": raw_incident_data.get('raw_content', {}).get('summary'),
                    "link": raw_incident_data.get('raw_content', {}).get('link'),
                    "timestamp": raw_incident_data.get('timestamp'),
                    "location": raw_incident_data.get('raw_content', {}).get('location'),
                    "media_url": raw_incident_data.get('raw_content', {}).get('media_url')
                }
                
                if new_raw_content_entry not in existing_raw_content_list:
                    existing_raw_content_list.append(new_raw_content_entry)
                
                texts_for_resummary = [item.get('summary', item.get('title', '')) for item in existing_raw_content_list if item.get('summary') or item.get('title')]
                if incident_text not in texts_for_resummary: # Ensure current incident's text is included
                     texts_for_resummary.append(incident_text) 

                resummary_result = re_summarize_incident_text_func(texts_for_resummary, location_hint) 
                
                if resummary_result.get("status") == "success":
                    updated_summary = resummary_result.get("summary", existing_metadata.get("original_summary_text", incident_text))
                    updated_category = resummary_result.get("category", existing_metadata.get("category", category))
                    updated_prediction = resummary_result.get("prediction", "")
                else:
                    print(f"[check_dedup] WARNING: Re-summarization failed. Using fallback summary/category.")
                    updated_summary = f"{existing_metadata.get('original_summary_text', '')} {incident_text}".strip()
                    updated_category = existing_metadata.get("category", category)
                    updated_prediction = existing_metadata.get("prediction", "")
                
                # Create the updated incident payload for Vector Search metadata
                updated_metadata_dict = {
                    "original_summary_text": updated_summary,
                    "source_type": existing_metadata.get("source_type", source_type), 
                    "category": updated_category,
                    "prediction": updated_prediction,
                    "timestamp_raw": existing_metadata.get("timestamp_raw", raw_incident_data.get('timestamp')),
                    "location_hint": existing_metadata.get("location_hint", location_hint),
                    "raw_content_updates": existing_raw_content_list, 
                    "event_id": existing_vector_id 
                }
                updated_metadata_json_str = json.dumps(updated_metadata_dict)

                # Generate new embedding for the updated summary
                updated_embedding_response = embedding_model.get_embeddings([updated_summary])
                raw_updated_embedding = np.array(updated_embedding_response[0].values)

                # --- CRITICAL CORRECTION: Perform L2 Normalization for UPDATED embedding ---
                updated_norm = np.linalg.norm(raw_updated_embedding)
                if updated_norm == 0:
                    updated_embedding = raw_updated_embedding.tolist()
                else:
                    updated_embedding = (raw_updated_embedding / updated_norm).tolist()
                
                # Create the updated datapoint for upsert
                updated_datapoint = aiplatform_v1.IndexDatapoint( 
                    datapoint_id=existing_vector_id, # Update the EXISTING ID
                    feature_vector=updated_embedding,
                    numeric_restricts=[ 
                        aiplatform_v1.IndexDatapoint.NumericRestriction(
                            namespace="timestamp_processed",
                            value_int=int(datetime.now(timezone.utc).timestamp()) # Use value_int consistently
                        )
                    ],
                    restricts=[ 
                        aiplatform_v1.IndexDatapoint.Restriction(
                            namespace="source_type",
                            allow_list=[source_type]
                        ),
                        aiplatform_v1.IndexDatapoint.Restriction(
                            namespace="category",
                            allow_list=[updated_category]
                        )
                    ],
                    metadata_json=updated_metadata_json_str
                )
                
                # Upsert to Vector Search using mutate_deployed_index
                upsert_request = aiplatform_v1.MutateDeployedIndexRequest(
                    index_endpoint=VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME,
                    deployed_index_id=VECTOR_SEARCH_DEPLOYED_INDEX_ID, # Use env var deployed index ID
                    upsert_datapoints=[updated_datapoint]
                )
                
                upsert_response = vector_search_client.mutate_deployed_index(upsert_request)
                print(f"[check_dedup] Successfully updated embedding and metadata for {existing_vector_id} in Vector Search.")
                
                # Return 'updated' status with the new payload for the agent to publish
                # Note: 'updated_incident_payload' should derive from the *updated_metadata_dict*
                updated_incident_payload = {
                    "event_id": existing_vector_id,
                    "original_source_id": existing_vector_id, 
                    "source_type": updated_metadata_dict.get("source_type"), 
                    "category": updated_category,
                    "summary": updated_summary,
                    "location": raw_incident_data.get('raw_content', {}).get('location', {}), # Current incident's location is a reasonable fallback
                    "location_hint": location_hint,
                    "timestamp_raw": updated_metadata_dict.get("timestamp_raw"),
                    "timestamp_processed": datetime.now(timezone.utc).isoformat(),
                    "predicted_impact": updated_prediction,
                    "media_url": raw_incident_data.get('raw_content', {}).get('media_url', ''), # Current incident's media is a reasonable fallback
                    "deduplication_key_used": existing_vector_id, 
                    "incident_embedding": updated_embedding, # Storing the embedding in structured_incident
                    "raw_content_updates": existing_raw_content_list 
                }
                return {"status": "updated", "updated_payload": updated_incident_payload}
                
            except Exception as merge_error:
                print(f"[check_dedup] Error during incident merging: {merge_error}")
                # CRITICAL FIX: Return error status, not 'new'
                return {"status": "error", "message": f"Merge failed: {merge_error}"}
                
        else: # No match found in Vector Search, so it's a new incident
            print(f"[check_dedup] No semantic match found for {event_id}. Upserting new incident.")
            
            # Prepare metadata for the new incident
            new_incident_metadata = {
                "event_id": event_id,
                "original_source_id": event_id,
                "source_type": source_type,
                "category": category,
                "original_summary_text": incident_text,
                "location": raw_incident_data.get('raw_content', {}).get('location', {}),
                "location_hint": location_hint,
                "timestamp_raw": raw_incident_data.get('timestamp'),
                "timestamp_processed": int(datetime.now(timezone.utc).timestamp()), # Store as int timestamp for consistency
                "media_url": raw_incident_data.get('raw_content', {}).get('media_url', ''),
                "raw_content_updates": [{ # Store original raw content as a list
                    "source": raw_incident_data.get('source'),
                    "title": raw_incident_data.get('raw_content', {}).get('title'),
                    "summary": raw_incident_data.get('raw_content', {}).get('summary'),
                    "link": raw_incident_data.get('raw_content', {}).get('link'),
                    "timestamp": raw_incident_data.get('timestamp'),
                    "location": raw_incident_data.get('raw_content', {}).get('location'),
                    "media_url": raw_incident_data.get('raw_content', {}).get('media_url')
                }]
            }
            # Convert metadata to JSON string
            new_metadata_json_str = json.dumps(new_incident_metadata)
            
            # Create the datapoint for the new incident
            new_incident_embedding_response = embedding_model.get_embeddings([incident_text])
            raw_new_embedding = np.array(new_incident_embedding_response[0].values)

            # --- CRITICAL CORRECTION: Perform L2 Normalization for NEW embedding ---
            new_norm = np.linalg.norm(raw_new_embedding)
            if new_norm == 0:
                new_incident_embedding = raw_new_embedding.tolist()
            else:
                new_incident_embedding = (raw_new_embedding / new_norm).tolist()

            new_datapoint = aiplatform_v1.IndexDatapoint( # Use aiplatform_v1.IndexDatapoint for upsert
                datapoint_id=event_id, 
                feature_vector=new_incident_embedding,
                numeric_restricts=[ 
                    aiplatform_v1.IndexDatapoint.NumericRestriction(
                        namespace="timestamp_processed",
                        value_int=new_incident_metadata["timestamp_processed"] # Use value_int for timestamp consistency
                    )
                ],
                restricts=[ 
                    aiplatform_v1.IndexDatapoint.Restriction(
                        namespace="source_type",
                        allow_list=[source_type]
                    ),
                    aiplatform_v1.IndexDatapoint.Restriction(
                        namespace="category",
                        allow_list=[category]
                    )
                ],
                metadata_json=new_metadata_json_str
            )
            
            # Upsert to Vector Search using mutate_deployed_index
            upsert_request = aiplatform_v1.MutateDeployedIndexRequest(
                index_endpoint=VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME,
                deployed_index_id=VECTOR_SEARCH_DEPLOYED_INDEX_ID, # Use env var deployed index ID
                upsert_datapoints=[new_datapoint]
            )
            
            upsert_response = vector_search_client.mutate_deployed_index(request=upsert_request)
            print(f"[check_dedup] Successfully stored new incident {event_id} in Vector Search.")
            return {"status": "new", "original_incident_id": event_id, "new_payload": new_metadata_json_str}

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