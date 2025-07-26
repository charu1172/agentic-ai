# # tools.py
# import os # Added for os.environ.get
# import json
# import requests # Added for requests.get
# import feedparser
# from datetime import datetime, timezone

# # Correct import for Tool decorator from google.adk
# from google.adk.tools import Tool 
# from google.cloud import pubsub_v1 # Google Cloud Python ADK for Pub/Sub


# # --- Global Configurations for Tools ---
# # Use os.environ.get to get PROJECT_ID from environment variables set during deployment.
# PROJECT_ID = os.environ.get("GCP_PROJECT_ID") 
# if not PROJECT_ID:
#     raise ValueError("GCP_PROJECT_ID environment variable not set. Cannot initialize Pub/Sub client for tools.")

# # This should be just the topic NAME, not a path
# PUBSUB_TOPIC_NAME = "raw-incident-feed" 

# # Initialize Pub/Sub publisher client globally
# publisher = pubsub_sub_v1.PublisherClient()
# # Construct the full topic path dynamically using the PROJECT_ID
# PUB_SUB_TOPIC_PATH_STRING = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC_NAME)


# # --- Tool Definition: fetch_rss ---
# @Tool.define(name="fetch_rss", description="Fetch news from local RSS feed.")
# def fetch_rss() -> dict: # Function should ideally take parameters if the agent is to provide them
#     RSS_FEED_URL = "https://www.thehindu.com/news/cities/bangalore/?service=rss" # Hardcoded URL, consider passing as param
#     print(f"[fetch_rss] Executing for URL: {RSS_FEED_URL}")
#     try:
#         res = requests.get(RSS_FEED_URL)
#         res.raise_for_status()
#         feed = feedparser.parse(res.text)

#         entries = []
#         for entry in feed.entries[:5]: # Limit to recent 5 for demo/speed
#             entries.append({
#                 "title": entry.get("title", ""),
#                 "link": entry.get("link", ""),
#                 "summary": entry.get("summary", ""),
#                 "published_date": entry.get("published", "") # Use consistent key 'published_date'
#             })
#         return {"status": "success", "entries": entries}
#     except Exception as e:
#         print(f"[fetch_rss] Error: {e}")
#         return {"status": "error", "message": str(e)}

# # --- Tool Definition: publish_incident ---
# @Tool.define(name="publish_incident", description="Publish formatted incident to Pub/Sub.")
# def publish_to_pubsub(incident: dict) -> dict: # Takes a dict, returns a dict with status/message_id
#     """
#     Publishes a single standardized incident to the raw-incident-feed Pub/Sub topic.
#     The incident dict should contain 'source', 'raw_content' (including title, link, etc.),
#     and other optional fields like 'location_hint' and 'timestamp'.
#     Returns a dictionary with 'status' and 'message_id' on success.
#     """
#     print(f"[publish_incident] Attempting to publish incident.")
#     try:
#         # Add timestamp and location_hint if not already present in the incident dict
#         if "timestamp" not in incident:
#             incident["timestamp"] = datetime.now(timezone.utc).isoformat()
#         if "location_hint" not in incident:
#             incident["location_hint"] = "Bengaluru" # Default location hint

#         message_bytes = json.dumps(incident).encode("utf-8")
        
#         # Use the dynamically constructed topic_path string
#         future = publisher.publish(PUB_SUB_TOPIC_PATH_STRING, message_bytes) 
#         message_id = future.result() # This will block until publish is confirmed or errors
        
#         print(f"[publish_incident] Published message ID: {message_id} for: {incident.get('raw_content', {}).get('title', 'No Title')}")
#         return {"status": "success", "message_id": message_id}
#     except Exception as e:
#         print(f"[publish_incident] Error publishing: {e}")
#         return {"status": "error", "message": str(e)}