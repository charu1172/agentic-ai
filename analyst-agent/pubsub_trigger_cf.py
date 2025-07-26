import os
import json
import base64
import requests
import functions_framework

# --- Configuration for CF Wrapper ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")

# --- IMPORTANT: URL of your deployed Cloud Run Analyst Agent service ---
# After deploying the Analyst Agent to Cloud Run, get its service URL.
# Example: ANALYST_AGENT_CLOUD_RUN_URL = "https://city-pulse-analyst-agent-xxxxx-el.a.run.app"
ANALYST_AGENT_CLOUD_RUN_URL = os.environ.get(
    "ANALYST_AGENT_CLOUD_RUN_URL",
    "https://city-pulse-analyst-agent-557703177809.asia-south1.run.app"
)

if not ANALYST_AGENT_CLOUD_RUN_URL:
    raise ValueError("Missing ANALYST_AGENT_CLOUD_RUN_URL environment variable for CF wrapper.")

# --- Cloud Function Entry Point (for HTTP trigger) ---
@functions_framework.http
def analyst_agent_http_trigger(request):
    """
    Cloud Function triggered by HTTP requests (can be called from Pub/Sub push subscription).
    This function acts as a wrapper to invoke the Analyst Agent deployed on Cloud Run.
    """
    # Generate correlation ID for tracking
    import uuid
    correlation_id = str(uuid.uuid4())[:8]
    print(f"[{correlation_id}] [CF_Wrapper] Cloud Function (Analyst Agent HTTP Trigger) activated.")

    # Extract data from HTTP request (Pub/Sub push subscription format)
    try:
        # Get the request JSON data
        request_json = request.get_json()
        if not request_json:
            print(f"[{correlation_id}] [CF_Wrapper] Received empty HTTP request. Skipping.")
            return "OK", 200

        # Extract Pub/Sub message from push subscription format
        message = request_json.get("message", {})
        message_data = message.get("data", "")
        
        if not message_data:
            print(f"[{correlation_id}] [CF_Wrapper] Received empty Pub/Sub message. Skipping.")
            return "OK", 200
        # 1. Decode Pub/Sub Message Payload
        json_string = base64.b64decode(message_data).decode('utf-8')
        raw_incident_data_from_pubsub = json.loads(json_string)
        
        # Add the event ID to the incident data
        raw_incident_data_from_pubsub["pubsub_event_id"] = correlation_id

        print(f"[{correlation_id}] [CF_Wrapper] Sending data to Cloud Run Analyst Agent: {raw_incident_data_from_pubsub.get('source')} - {raw_incident_data_from_pubsub.get('raw_content',{}).get('title', 'N/A')}")
        
        # 2. Make an HTTP POST request to the Cloud Run service endpoint
        headers = {"Content-Type": "application/json"}
        endpoint_url = f"{ANALYST_AGENT_CLOUD_RUN_URL}/process-incident"
        
        # Send the incident data directly (not wrapped in raw_incident)
        response = requests.post(
            endpoint_url, 
            headers=headers, 
            json=raw_incident_data_from_pubsub,  # Send data directly
            timeout=300  # 5 minute timeout for processing
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        print(f"[{correlation_id}] [CF_Wrapper] Cloud Run Analyst Agent responded: {response.status_code} - {response.text}")

    except json.JSONDecodeError as e:
        print(f"[{correlation_id}] [CF_Wrapper] ERROR: JSON Decode Error in CF wrapper: {e}. Raw data: {request.get_data()}")
        # Return error for HTTP function
        return f"JSON Decode Error: {e}", 400
    except requests.exceptions.RequestException as req_e:
        print(f"[{correlation_id}] [CF_Wrapper] ERROR: HTTP Request Error calling Cloud Run Agent: {req_e}")
        # Return error for HTTP function
        return f"HTTP Request Error: {req_e}", 500
    except Exception as e:
        print(f"[{correlation_id}] [CF_Wrapper] CRITICAL ERROR in CF wrapper: {e}. Request: {request.get_data()}")
        return f"Critical Error: {e}", 500

    print(f"[{correlation_id}] [CF_Wrapper] Cloud Function (Analyst Agent HTTP Trigger) finished.")
    return "OK", 200