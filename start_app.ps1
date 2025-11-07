# Quick Start Script for Windows PowerShell
# Run this script to start the Streamlit app

Write-Host "🌤️  Starting Guwahati Weather Prediction System..." -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is installed
try {
    $streamlitVersion = streamlit --version
    Write-Host "✅ Streamlit is installed: $streamlitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Streamlit is not installed!" -ForegroundColor Red
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "🚀 Launching application..." -ForegroundColor Cyan
Write-Host "📱 The app will open in your default browser" -ForegroundColor Yellow
Write-Host "🔗 URL: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run streamlit
streamlit run app.py
