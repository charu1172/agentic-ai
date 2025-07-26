import os 
import sys
from datetime import datetime, timezone 
from google.adk import Agent

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

from tools import (
    check_and_record_semantic_deduplication,
    analyze_with_gemini, 
    publish_structured_incident,
    re_summarize_incident_text 
)

# This is the required variable name for ADK Cloud Run deployment
root_agent = Agent( 
    name="city_pulse_analyst_agent",
    model="gemini-1.5-flash",  
    instructions=(
        "You are an autonomous Analyst Agent. Your primary goal is to process incoming raw urban incident data. "
        "Each raw incident is a JSON object provided as your input stimulus, coming from the 'raw-incident-feed' Pub/Sub topic. "
        "You must follow these steps precisely:\n\n"
        "1.  **Parse Input**: Expect your input to be a JSON string representing a raw incident. Parse it carefully to extract 'raw_incident' and 'event_id'. "
        "    From 'raw_incident', extract 'source', 'raw_content' (which contains 'title', 'summary', 'text_description'), "
        "    'timestamp' (from raw_incident), 'location_hint', and 'media_url' (if available)."
        "    The primary text for analysis should be raw_content['summary'] or raw_content['title'] or raw_content['text_description'].\n"
        "2.  **Semantic Deduplication & Enhancement**: Use the 'check_and_record_semantic_deduplication' tool. "
        "    - Pass the most descriptive text (summary/title/description from raw_content) as 'incident_text'. "
        "    - Pass 'event_id', 'source' (as source_type), 'category' (you can use 'General' initially), and the entire 'raw_incident' object (as raw_incident_data) to the tool."
        "    - Also pass 'location_hint' from the raw_incident."
        "    - **Critical Decision (Based on Tool Status)**:\n"
        "        - **If tool returns 'status: new'**: Use the 'new_payload' for final structured incident. Continue to step 3 for additional analysis.\n"
        "        - **If tool returns 'status: updated'**: Use the 'updated_payload' for final structured incident. Skip step 3 and 4 (re-summarization already done internally). Go directly to step 5.\n"
        "        - **If tool returns 'status: error'**: Report 'Semantic deduplication failed: [error message]. Proceeding with analysis but without deduplication.' Continue to step 3 but use original incident data.\n"
        "        - **If tool returns 'status: skipped_no_vector_db' or 'status: skipped_no_text'**: Continue to step 3 with original incident data.\n\n"
        "3.  **Analyze Incident**: Only if the incident is 'new' or deduplication was skipped/failed, use the 'analyze_with_gemini' tool. "
        "    - Pass the primary incident text (from raw_content) to this tool for analysis.\n"
        "    - Combine the analysis results with the original incident data to create a structured incident payload.\n"
        "4.  **Skip Re-summarization**: The 're_summarize_incident_text' tool is called internally by the deduplication tool when needed. Do not call it separately.\n"
        "5.  **Publish Structured Data**: Use the 'publish_structured_incident' tool to publish the final structured incident to the 'structured-incident-feed' topic. "
        "    - For 'updated' incidents: Use the 'updated_payload' from step 2 directly.\n"
        "    - For 'new' incidents: Combine original incident data with analysis results from step 3.\n"
        "    - For error cases: Create a structured incident with available data and error details.\n\n"
        "**Important Guidelines**:\n"
        "- Always use JSON parsing to extract data from your input.\n"
        "- Handle errors gracefully and report them clearly.\n"
        "- The deduplication tool handles re-summarization internally for 'updated' incidents.\n"
        "- Include original raw data in your final structured output for traceability.\n"
        "- Be thorough in error reporting and include context about what went wrong."
    ),
    tools=[
        check_and_record_semantic_deduplication,
        analyze_with_gemini,
        publish_structured_incident,
        re_summarize_incident_text
    ]
)
