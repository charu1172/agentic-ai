import os
import json
import logging
import traceback
import asyncio
from flask import Flask, request, jsonify
from google.adk.core import Context
from google.adk.core.message import Message, Part

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app instance
app = Flask(__name__)

# Global ADK agent instance - initialized once when module loads
agent = None

def initialize_agent():
    global agent  # Ensure we're modifying the module-level 'agent' variable
    try:
        logger.info("Initializing ADK Agent...")

        # Import your tool instances here
        from tools import (
            re_summarize_incident_text,
            check_and_record_semantic_deduplication,
            analyze_with_gemini,
            publish_structured_incident,
        )
        logger.info("Tools imported successfully")

        from google.adk import Agent
        logger.info("ADK imports successful")

        # Initialize ADK Agent
        logger.info("Creating Agent instance...")
        try:
            agent = Agent(
                name="city_pulse_analyst_agent",
                model="gemini-1.5-flash",
                tools=[
                    re_summarize_incident_text,
                    check_and_record_semantic_deduplication,
                    analyze_with_gemini,
                    publish_structured_incident,
                ],
            )
            logger.info("Agent instance created successfully!")
            logger.info("ADK Agent initialized successfully!")
            return True
        except Exception as agent_error:
            logger.error(f"CRITICAL: Failed to create Agent instance: {agent_error}")
            logger.error(f"Agent creation traceback: {traceback.format_exc()}")
            return False

    except ImportError as ie:
        logger.error(f"Import error during agent initialization: {ie}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize ADK Agent: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Initialize agent eagerly at module load (important for Cloud Run production!)
logger.info("Calling agent initialization during module load...")
try:
    agent_initialized = initialize_agent()
    if agent_initialized:
        logger.info("SUCCESS: Agent initialized during module load!")
    else:
        logger.error("FAILED: Agent initialization returned False during module load")
except Exception as module_load_error:
    logger.error(f"EXCEPTION during module load agent initialization: {module_load_error}")
    logger.error(f"Module load exception traceback: {traceback.format_exc()}")

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "city-pulse-analyst-agent",
            "version": "1.0.0",
            "agent_initialized": agent is not None,
        }
    )

@app.route("/ready", methods=["GET"])
def readiness_check():
    """Readiness check endpoint."""
    if agent is None:
        return (
            jsonify({"status": "not_ready", "message": "Agent not initialized"}),
            503,
        )

    return jsonify({"status": "ready", "service": "city-pulse-analyst-agent"})

@app.route("/debug", methods=["GET"])
def debug_status():
    """Debug endpoint to check service status."""
    try:
        env_vars = {
            "GCP_PROJECT_ID": os.environ.get("GCP_PROJECT_ID"),
            "GCP_REGION": os.environ.get("GCP_REGION"),
            "PORT": os.environ.get("PORT"),
        }

        # Check if tools can be imported
        tools_status = "unknown"
        try:
            from tools import (
                re_summarize_incident_text,
                check_and_record_semantic_deduplication,
                analyze_with_gemini,
                publish_structured_incident,
            )

            tools_status = "imported_successfully"
        except Exception as te:
            tools_status = f"import_failed: {str(te)}"

        # Check if ADK can be imported
        adk_status = "unknown"
        try:
            from google.adk import Agent

            adk_status = "imported_successfully"
        except Exception as ae:
            adk_status = f"import_failed: {str(ae)}"

        # Try to initialize agent if not already done
        agent_init_error = None
        if agent is None:
            try:
                logger.info("Debug endpoint: Attempting agent initialization...")
                result = initialize_agent()
                if not result:
                    agent_init_error = "initialization_returned_false"
            except Exception as e:
                agent_init_error = f"initialization_exception: {str(e)}"

        return jsonify(
            {
                "agent_initialized": agent is not None,
                "environment_variables": env_vars,
                "tools_import_status": tools_status,
                "adk_import_status": adk_status,
                "agent_initialization_error": agent_init_error,
                "service": "city-pulse-analyst-agent",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Debug check failed: {str(e)}"}), 500

@app.route("/process-incident", methods=["POST"])
def process_incident():
    """Process a raw incident through the ADK agent."""
    logger.info("process-incident endpoint called")

    if agent is None:
        logger.error("Agent not initialized - returning 503")
        return jsonify({"error": "Agent not initialized"}), 503

    try:
        incident_data = request.get_json()

        if not incident_data:
            logger.error("No incident data provided")
            return jsonify({"error": "No incident data provided"}), 400

        logger.info(f"Processing incident: {incident_data.get('event_id', 'unknown')}")

        incident_text = json.dumps(incident_data, indent=2)
        prompt = f"Process this raw incident:\n{incident_text}"

        # Create proper context and message
        context = Context(agent=agent)
        message = Message(parts=[Part(text=prompt)])
        
        # run_live returns an async generator, we need to collect all responses
        async def collect_responses():
            responses = []
            async for response in agent.run_live(message, context=context):
                responses.append(str(response))
            return responses
        
        # Run the async function
        responses = asyncio.run(collect_responses())
        response = "\n".join(responses) if responses else "No response"

        logger.info(f"Successfully processed incident: {incident_data.get('event_id', 'unknown')}")

        return jsonify(
            {
                "status": "success",
                "incident_id": incident_data.get("event_id", "unknown"),
                "response": str(response),
            }
        )

    except Exception as e:
        logger.error(f"Error processing incident: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/pubsub-push", methods=["POST"])
def pubsub_push():
    """Handle Pub/Sub push messages directly."""
    if agent is None:
        return jsonify({"error": "Agent not initialized"}), 503

    try:
        envelope = request.get_json()

        if not envelope:
            return jsonify({"error": "No Pub/Sub data"}), 400

        pubsub_message = envelope.get("message", {})
        message_data = pubsub_message.get("data", "")

        import base64

        decoded_data = base64.b64decode(message_data).decode("utf-8")
        incident_data = json.loads(decoded_data)

        incident_text = json.dumps(incident_data, indent=2)
        prompt = f"Process this raw incident:\n{incident_text}"

        # Create proper context and message
        context = Context(agent=agent)
        message = Message(parts=[Part(text=prompt)])
        
        # run_live returns an async generator, we need to collect all responses
        async def collect_responses():
            responses = []
            async for response in agent.run_live(message, context=context):
                responses.append(str(response))
            return responses
        
        # Run the async function
        responses = asyncio.run(collect_responses())
        response = "\n".join(responses) if responses else "No response"

        logger.info(f"Successfully processed Pub/Sub incident: {incident_data.get('event_id', 'unknown')}")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        logger.error(f"Error processing Pub/Sub message: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # For local development only: use flask's built-in server
    port = int(os.environ.get("PORT", 8080))

    logger.info(f"Starting City Pulse Analyst Agent on port {port} (local development mode)")
    logger.info("Environment check:")
    logger.info(f"GCP_PROJECT_ID: {os.environ.get('GCP_PROJECT_ID')}")
    logger.info(f"GCP_REGION: {os.environ.get('GCP_REGION')}")
    logger.info(f"Agent already initialized: {agent is not None}")

    logger.info(f"Starting Flask dev server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
