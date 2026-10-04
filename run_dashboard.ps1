# PowerShell launcher for Billet Inspection Dashboard
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Automated Billet Quality Inspection System Dashboard" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Streamlit web server..." -ForegroundColor Yellow
python -m streamlit run app.py
