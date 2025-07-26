# Cloud Run Deployment Script for City Pulse Analyst Agent
# Set environment variables and deploy using ADK CLI

# Verify ADK CLI installation
Write-Host "Verifying ADK CLI installation..."
try {
    $adkVersion = adk --version
    Write-Host "ADK CLI version: $adkVersion"
} catch {
    Write-Error "ADK CLI not found. Please install it first: https://github.com/googleapis/agent-development-kit"
    exit 1
}

# Set your Google Cloud project and region
$env:GOOGLE_CLOUD_PROJECT="tidy-computing-466411-j9"
$env:GOOGLE_CLOUD_LOCATION="asia-south1"
$env:GOOGLE_GENAI_USE_VERTEXAI="True"

# ADK deployment environment variables (will be passed to Cloud Run)
$env:GCP_PROJECT_ID="tidy-computing-466411-j9"
$env:GCP_REGION="asia-south1"

# Vector Search environment variables (updated with your actual values)
$env:VECTOR_SEARCH_API_ENDPOINT="1645951261.asia-south1-557703177809.vdb.vertexai.goog"
$env:VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME="projects/557703177809/locations/asia-south1/indexEndpoints/5643872350211407872"
$env:VECTOR_SEARCH_DEPLOYED_INDEX_ID="9176946257883561984"

Write-Host "Environment variables set:"
Write-Host "  GOOGLE_CLOUD_PROJECT: $env:GOOGLE_CLOUD_PROJECT"
Write-Host "  GOOGLE_CLOUD_LOCATION: $env:GOOGLE_CLOUD_LOCATION"
Write-Host "  Vector Search API Endpoint: $env:VECTOR_SEARCH_API_ENDPOINT"
Write-Host "  Vector Search Resource Name: $env:VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME"
Write-Host "  Vector Search Deployed Index ID: $env:VECTOR_SEARCH_DEPLOYED_INDEX_ID"

# Deploy using ADK CLI (environment variables will be inherited from current shell)
Write-Host ""
Write-Host "Starting ADK Cloud Run deployment..."

adk deploy cloud_run `
  --project=$env:GOOGLE_CLOUD_PROJECT `
  --region=$env:GOOGLE_CLOUD_LOCATION `
  --service_name="city-pulse-analyst-agent" `
  --app_name="analyst-agent" `
  --with_ui `
  --allow_unauthenticated `
  .

Write-Host ""
Write-Host "Deployment completed! Check the output above for the service URL."
Write-Host ""
Write-Host "Important Notes:"
Write-Host "1. Ensure your Cloud Run service account has the following IAM roles:"
Write-Host "   - Vertex AI User (for Gemini and embeddings)"
Write-Host "   - Pub/Sub Publisher (for publishing structured incidents)" 
Write-Host "   - Vertex AI Vector Search Editor (for semantic deduplication)"
Write-Host "   - Firestore User (if using Firestore for additional storage)"
Write-Host ""
Write-Host "2. Your Vector Search configuration:"
Write-Host "   - Cross-project setup: Resources in 557703177809, deployed in tidy-computing-466411-j9"
Write-Host "   - API Endpoint: $env:VECTOR_SEARCH_API_ENDPOINT"
Write-Host "   - Resource Name: $env:VECTOR_SEARCH_ENDPOINT_RESOURCE_NAME" 
Write-Host "   - Deployed Index ID: $env:VECTOR_SEARCH_DEPLOYED_INDEX_ID"
