# Production Setup

## Streamlit Server
Run the Streamlit application in a Docker container:

```bash
docker build -t patient-risk-dashboard:latest .
docker run -p 8501:8501 patient-risk-dashboard:latest
```

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)