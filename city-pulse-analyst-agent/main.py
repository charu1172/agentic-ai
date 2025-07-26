import os
import vertexai
from vertexai.preview import reasoning_engines
from vertexai import agent_engines

from analyst_agent import analyst_agent

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") 
GCP_REGION = os.environ.get("GCP_REGION")

if not PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID environment variable not set. Please set it in your terminal (e.g., export GCP_PROJECT_ID='your-project-id').")
if not GCP_REGION:
    raise ValueError("GCP_REGION environment variable not set. Please set it in your terminal (e.g., export GCP_REGION='asia-south1').")

vertexai.init(project=PROJECT_ID, location=GCP_REGION, staging_bucket=f"gs://{PROJECT_ID}-staging")

# --- Define the AdkApp for Deployment ---
# This defines your agent application within the Vertex AI Agent Engine framework.
app = reasoning_engines.AdkApp(
    agent=analyst_agent,
    enable_tracing=True,
)

# --- Deployment Call ---
# This part of the script runs when you execute 'python main.py' locally.
if __name__ == "__main__":
    print(f"Deploying Analyst Agent to Vertex AI Agent Engine in project {PROJECT_ID} in {GCP_REGION}...")
    try:
        # Use the NEW agent_engines.create() method from the official documentation
        deployed_agent = agent_engines.create(
            agent_engine=app,
            requirements=[
                "google-cloud-aiplatform[adk,agent_engines]",
                "google-cloud-pubsub",
                "google-cloud-firestore",
                "cloudpickle"
            ]
        )
        
        print(f"\n🎉 Analyst Agent deployed successfully!")
        print(f"📋 Resource name: {deployed_agent.resource_name}")
        print(f"🌐 View in console: https://console.cloud.google.com/vertex-ai/agents/agent-engines?project={PROJECT_ID}")
        
        # Test the deployed agent
        print(f"\n🧪 Testing deployed agent...")
        session = deployed_agent.create_session(user_id="test_user")
        print(f"✅ Session created: {session['id']}")
        
        print(f"\n📝 Agent is ready to process incidents from raw-incident-feed!")
        print(f"📊 Monitor at: https://console.cloud.google.com/vertex-ai/agents/agent-engines")
        
    except Exception as e:
        print(f"\n❌ Error deploying Analyst Agent: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Ensure you have Agent Engine permissions in {GCP_REGION}")
        print(f"   2. Check if {GCP_REGION} supports Agent Engine")
        print(f"   3. Try deploying to us-central1 if asia-south1 has issues")
        raise