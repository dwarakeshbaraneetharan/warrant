#!/bin/bash
set -e

echo "========================================="
echo "Warrant - Google Cloud Run Deploy"
echo "========================================="
echo ""

if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI (gcloud) is not installed."
    echo "Please install it using Homebrew:"
    echo "  brew install --cask google-cloud-sdk"
    echo ""
    echo "Then authenticate and set your project:"
    echo "  gcloud auth login"
    echo "  gcloud config set project YOUR_PROJECT_ID"
    echo "  gcloud auth configure-docker"
    echo ""
    echo "Once that's done, re-run this script!"
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No Google Cloud project configured."
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "✅ Using GCP Project: $PROJECT_ID"
echo ""

read -p "Enter your Groq API Key: " GROQ_API_KEY
if [ -z "$GROQ_API_KEY" ]; then
    echo "Groq API Key is required!"
    exit 1
fi

echo ""
echo "🚀 Building and deploying to Cloud Run..."
echo "This will take a few minutes as Cloud Build packages the Docker container."
echo ""

gcloud run deploy warrant \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --cpu-boost \
  --no-cpu-throttling \
  --timeout 300s \
  --set-env-vars "GROQ_API_KEY=${GROQ_API_KEY}" \
  --quiet

echo ""
echo "✅ Deployment complete!"
echo "Your app is now live on Google Cloud Run."
