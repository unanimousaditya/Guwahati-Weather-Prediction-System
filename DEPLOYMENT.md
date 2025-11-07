# 🚀 Deployment Guide for Guwahati Weather Predictor

## Option 1: Run Locally (Easiest)

### Windows:

1. Open PowerShell in this folder
2. Run the start script:
   ```powershell
   .\start_app.ps1
   ```
   OR manually:
   ```powershell
   pip install -r requirements.txt
   streamlit run app.py
   ```

### Mac/Linux:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## Option 2: Deploy to Streamlit Cloud (Free & Recommended)

### Steps:

1. **Create GitHub Account** (if you don't have one)

   - Go to https://github.com
   - Sign up for free

2. **Create a New Repository**

   - Click "New Repository"
   - Name it: `guwahati-weather-predictor`
   - Make it Public
   - Don't initialize with README (we already have one)

3. **Upload Your Files**

   Option A - Using GitHub Desktop (Easier):

   - Download GitHub Desktop
   - Clone your new repository
   - Copy all files from this folder to the cloned repository
   - Commit and push

   Option B - Using Git Command Line:

   ```bash
   cd "C:\Users\Admin\Downloads\Guwahati_weather_1973-2023"
   git init
   git add .
   git commit -m "Initial commit: Weather prediction app"
   git remote add origin https://github.com/YOUR_USERNAME/guwahati-weather-predictor.git
   git push -u origin main
   ```

4. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "Sign in" (use your GitHub account)
   - Click "New app"
   - Select:
     - Repository: `YOUR_USERNAME/guwahati-weather-predictor`
     - Branch: `main`
     - Main file path: `app.py`
   - Click "Deploy!"
5. **Wait 2-5 minutes** for deployment to complete

6. **Your app is live!**
   - You'll get a URL like: `https://YOUR_USERNAME-guwahati-weather.streamlit.app`
   - Share it with anyone!

### Important Notes for Streamlit Cloud:

- ✅ Free tier includes: 1GB RAM, 1 CPU
- ✅ Your app will sleep after 7 days of inactivity
- ✅ All model files must be in the repository
- ⚠️ Large files (>100MB) may need Git LFS

---

## Option 3: Deploy to Heroku

1. **Install Heroku CLI**

   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**

   ```bash
   heroku login
   heroku create guwahati-weather-predictor
   ```

3. **Create Procfile**

   ```
   web: streamlit run app.py --server.port $PORT
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

---

## Option 4: Deploy to AWS/Azure/GCP

### Using Docker:

1. **Create Dockerfile**:

   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**:

   ```bash
   docker build -t weather-app .
   docker run -p 8501:8501 weather-app
   ```

3. **Deploy to Cloud**:
   - AWS: Elastic Beanstalk or ECS
   - Azure: Container Instances or App Service
   - GCP: Cloud Run or App Engine

---

## Troubleshooting

### Issue: "Module not found"

**Solution**:

```bash
pip install -r requirements.txt
```

### Issue: "Port already in use"

**Solution**:

```bash
streamlit run app.py --server.port 8502
```

### Issue: Model files too large for GitHub

**Solution**: Use Git LFS

```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.h5"
git add .gitattributes
git commit -m "Track large files with Git LFS"
```

### Issue: App runs but doesn't load data

**Solution**: Make sure all CSV files are in the same directory as app.py

### Issue: Out of memory on Streamlit Cloud

**Solution**:

- Reduce the number of historical years loaded
- Use data sampling for visualizations
- Consider upgrading to Streamlit Cloud Pro

---

## Performance Tips

1. **Optimize Data Loading**:

   - Load only necessary columns
   - Sample large datasets
   - Use caching effectively

2. **Reduce Model Size**:

   - Use model compression techniques
   - Consider using lighter models for deployment

3. **Improve Loading Speed**:
   - Use `@st.cache_data` and `@st.cache_resource`
   - Lazy load components
   - Optimize visualizations

---

## Monitoring & Maintenance

1. **Check App Health**:

   - Streamlit Cloud Dashboard
   - Monitor logs for errors
   - Test predictions regularly

2. **Update Models**:

   - Retrain with new data periodically
   - Replace `.pkl` files in repository
   - Redeploy app

3. **User Feedback**:
   - Add feedback form
   - Monitor usage statistics
   - Iterate based on user needs

---

## Security Considerations

1. **API Keys**: Use Streamlit secrets for sensitive data
2. **Rate Limiting**: Implement to prevent abuse
3. **Input Validation**: Sanitize user inputs
4. **HTTPS**: Always use secure connections
5. **Authentication**: Add if needed for sensitive deployments

---

## Cost Estimates

| Platform        | Free Tier      | Paid Plans     |
| --------------- | -------------- | -------------- |
| Streamlit Cloud | 1 app, 1GB RAM | $20-200+/month |
| Heroku          | Limited hours  | $7-500/month   |
| AWS             | 12 months free | $10-100+/month |
| Azure           | $200 credit    | $10-100+/month |
| GCP             | $300 credit    | $10-100+/month |

**Recommendation**: Start with **Streamlit Cloud (Free)** for testing and small deployments.

---

## Next Steps After Deployment

1. ✅ Test all features thoroughly
2. ✅ Share the URL with users
3. ✅ Monitor performance and errors
4. ✅ Collect user feedback
5. ✅ Plan for regular model updates
6. ✅ Consider adding authentication
7. ✅ Set up automated testing
8. ✅ Create user documentation

---

## Support

For issues or questions:

1. Check the README.md
2. Review Streamlit documentation: https://docs.streamlit.io
3. Open an issue on GitHub
4. Contact the developer

---

**Happy Deploying! 🚀**
