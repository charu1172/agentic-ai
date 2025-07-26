# Cloud Function Deployment Script for City Pulse Scout Agent
# Deploy RSS polling function to asia-south1 with public access

# Set your Google Cloud project and region
$env:GOOGLE_CLOUD_PROJECT="tidy-computing-466411-j9"
$env:GOOGLE_CLOUD_LOCATION="asia-south1"

Write-Host "Deploying City Pulse Scout Agent to Cloud Functions..."
Write-Host "Project: $env:GOOGLE_CLOUD_PROJECT"
Write-Host "Region: $env:GOOGLE_CLOUD_LOCATION"
Write-Host "Function: poll_news_rss"
Write-Host ""

# Deploy the RSS polling function with public access
gcloud functions deploy poll_news_rss `
  --gen2 `
  --runtime python311 `
  --region $env:GOOGLE_CLOUD_LOCATION `
  --source . `
  --entry-point poll_news_rss `
  --trigger-http `
  --allow-unauthenticated `
  --project $env:GOOGLE_CLOUD_PROJECT `
  --memory 256Mi `
  --timeout 540s `
  --max-instances 10

Write-Host ""
Write-Host "Deployment completed!"
Write-Host ""
Write-Host "Function URL will be displayed above."
Write-Host "The function is publicly accessible (--allow-unauthenticated)."
Write-Host ""
Write-Host "To test the function, you can:"
Write-Host "1. Visit the function URL in your browser"
Write-Host "2. Use curl: curl -X POST [FUNCTION_URL]"
Write-Host "3. Set up a Cloud Scheduler job to trigger it periodically"
Write-Host ""
Write-Host "Important Notes:"
Write-Host "1. Ensure your Cloud Function service account has:"
Write-Host "   - Pub/Sub Publisher role for publishing to raw-incident-feed topic"
Write-Host "2. The function will publish to topic: raw-incident-feed"
Write-Host "3. RSS source: https://www.thehindu.com/news/cities/bangalore/?service=rss"
