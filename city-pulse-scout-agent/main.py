import os
import json
import requests
import feedparser
from google.cloud import pubsub_v1
from datetime import datetime, timezone

PROJECT_ID = "tidy-computing-466411-j9"
RAW_INCIDENT_FEED_TOPIC_NAME = "raw-incident-feed" 
publisher = pubsub_v1.PublisherClient()
RAW_INCIDENT_FEED_TOPIC_PATH = publisher.topic_path(PROJECT_ID, RAW_INCIDENT_FEED_TOPIC_NAME)  

def publish_raw_incident(incident_data):
    """Publishes structured raw incident data to Pub/Sub."""
    try:
        message_json = json.dumps(incident_data)
        message_bytes = message_json.encode('utf-8')
        future = publisher.publish(RAW_INCIDENT_FEED_TOPIC_PATH, message_bytes)
        print(f"Published message ID: {future.result()}")
    except Exception as e:
        print(f"Error publishing message: {e}")
        raise # Re-raise to indicate failure

# --- Example: RSS Feed Poller Function ---
def poll_news_rss(request):

    print("Polling news RSS feed...")
    rss_url = "https://www.thehindu.com/news/cities/bangalore/?service=rss" 

    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            incident_data = {
                "source": "NewsRSS",
                "raw_content": {
                    "title": entry.get("title"),
                    "link": entry.get("link"),
                    "summary": entry.get("summary"),
                    "published_date": entry.get("published")
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location_hint": "Bengaluru"
            }
            print("--- Standardized Incident Data before Pub/Sub ---")
            print(json.dumps(incident_data, indent=2)) 
            print("-------------------------------------------------")

            publish_raw_incident(incident_data)
        print("Finished polling RSS feed.")
        return 'RSS polling complete', 200

    except requests.exceptions.RequestException as req_e:
        print(f"HTTP Request Error polling RSS feed: {req_e}")
        return f"HTTP Request Error: {req_e}", 500 
    except Exception as e:
        print(f"General Error polling RSS feed: {e}")
        return f"Internal Server Error: {e}", 500

def receive_user_report(request):
    """
    Cloud Function triggered by an HTTP POST request from the City Pulse WebApp.
    Handles user-submitted reports (text, media links).
    """
    print("Received user report...")
    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid JSON', 400

    user_id = request_json.get('user_id')
    raw_text = request_json.get('text_description')
    media_url = request_json.get('media_url') # Link to GCS where media was uploaded
    latitude = request_json.get('latitude')
    longitude = request_json.get('longitude')

    if not (user_id and (raw_text or media_url) and latitude is not None and longitude is not None):
        return 'Missing required fields (user_id, text/media, lat/long)', 400

    incident_data = {
        "source": "UserReport",
        "raw_content": {
            "user_id": user_id,
            "text_description": raw_text,
            "media_url": media_url,
            "location": {"latitude": latitude, "longitude": longitude}
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location_hint": f"{latitude},{longitude}" 
    }

    try:
        publish_raw_incident(incident_data)
        return 'Report received and published', 200
    except Exception as e:
        print(f"Failed to process user report: {e}")
        return 'Internal Server Error', 500

# implement similar functions for X/Reddit/EventBrite APIs based on their specific polling/webhook mechanisms