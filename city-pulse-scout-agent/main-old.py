# import os
# import vertexai
# from vertexai.preview import reasoning_engines

# from scout_agent import scout_agent
# # --- Vertex AI Initialization ---
# PROJECT_ID = os.environ.get("GCP_PROJECT_ID") 
# GCP_REGION = os.environ.get("GCP_REGION")

# if not PROJECT_ID:
#     raise ValueError("GCP_PROJECT_ID environment variable not set. Please set it in your terminal (e.g., export GCP_PROJECT_ID='your-project-id').")
# if not GCP_REGION:
#     raise ValueError("GCP_REGION environment variable not set. Please set it in your terminal (e.g., export GCP_REGION='asia-south1').")

# vertexai.init(project=PROJECT_ID, location=GCP_REGION)

# # --- Define the AdkApp for Deployment ---
# # This defines your agent application within the Vertex AI Reasoning Engines framework.
# app = reasoning_engines.AdkApp(
#     agent=scout_agent,
#     enable_tracing=True, 
# )

# # --- Deployment Call ---
# if __name__ == "__main__":
#     print(f"Deploying Scout Agent to Vertex AI Reasoning Engines in project {PROJECT_ID} in {GCP_REGION}...")
#     try:
#         # The 'deploy' method creates and deploys the agent instance.
#         # You might need to specify a service account explicitly if the default doesn't have permissions.
#         # Ensure this service account has Vertex AI User, Pub/Sub Publisher, etc. roles.
#         deployed_agent = app.deploy(
#             display_name="CityPulseScoutAgentInstance", # Name visible in Vertex AI Console
#             # service_account="YOUR_SERVICE_ACCOUNT_EMAIL@developer.gserviceaccount.com" # e.g., 557703177809-compute@developer.gserviceaccount.com
#             # IMPORTANT: Ensure this service account has necessary IAM roles: Vertex AI User, Pub/Sub Publisher.
#         )
#         print(f"\nScout Agent deployed successfully! Resource name: {deployed_agent.resource_name}")
#         print(f"View in console: https://console.cloud.google.com/vertex-ai/reasoning-engines/details/{deployed_agent.resource_name.split('/')[-1]}?project={PROJECT_ID}&region={GCP_REGION}")
#         print("\nTo trigger this agent, you would use: deployed_agent.run(input='{Your stimulus}') from another script.")
#         print("Or you can invoke it via the Vertex AI Console under Reasoning Engines (Test tab).")
#     except Exception as e:
#         print(f"\nError deploying Scout Agent: {e}")
#         raise