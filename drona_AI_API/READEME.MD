# Drona AI API - Setup Guide

## Setting Up the Virtual Environment

> **Note:** Your system must have Docker Desktop, Ollama with llama3.2:7B (4 GB Free RAM) and Postman for testing of API

```sh
cd drona_AI_API

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/Scripts/activate  

# Install dependencies
pip install -r requirements.txt

# Make an .env file
touch .env
add GROQ_API_KEY GOOGLE_API_KEY {Can be made in google console} 
```

---

## Setting Up the Database with Docker

```sh
docker run -d \                         
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

---

## Running the API

### Running Locally (Without Docker)

> **Note:** Remove the `host` parameter in the Ollama agent configuration before running this method.

```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> ⚠️ This method does not support the URL feature.

---

### Running with Docker

> **Important:**
> - Ensure the `host` parameter is set in the Ollama agent: `host="http://host.docker.internal:11434"`
> - The first startup may take some time.
> - Avoid modifying `requirements.txt` unnecessarily, as it will trigger a full reinstallation.

```sh
docker-compose up --build
```

