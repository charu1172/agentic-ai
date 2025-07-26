# import os 
# from datetime import datetime, timezone

# # Correct import for Agent class from google.adk
# from google.adk.agents import Agent 

# # Import the tools defined in tools.py
# from tools import fetch_rss, publish_to_pubsub 

# # --- Define Scout Agent Instance ---
# # This is the agent instance that will be deployed.
# scout_agent = Agent(
#     name="scout_agent",
#     model="gemini-1.5-flash", # Ensure this model is available in your region
#     description="An urban scout agent that fetches RSS feeds and publishes data to Pub/Sub.",
#     instruction=(
#         "You are a Scout Agent responsible for monitoring RSS feeds about city-level happenings. "
#         "Your primary goal is to continuously fetch the latest news using the 'fetch_rss' tool. "
#         "The 'fetch_rss' tool returns a JSON string with a 'status' and an 'entries' list. "
#         "If 'status' is 'success' and 'entries' is not empty, you MUST iterate through each entry. "
#         "For each news entry, you MUST standardize it into an incident format. "
#         "The standardized incident must be a JSON object with a 'source' (set to 'NewsRSS'), "
#         "and 'raw_content' (the original entry dictionary including 'title', 'link', 'summary', 'published_date'). "
#         "Then, you MUST publish each standardized incident individually to the City Pulse system "
#         "using the 'publish_incident' tool. "
#         "It's critical that 'publish_incident' is called for EACH individual incident after standardization. "
#         "If 'fetch_rss' returns an error, report the error. If no new entries, report 'No new entries to publish'."
#         "Always return a final summary of your actions after processing all entries."
#     ),
#     tools=[fetch_rss, publish_to_pubsub], # List the Tool objects (functions wrapped with @Tool.define)
#     enable_tool_calling=True # Added for explicit tool calling enablement
# )