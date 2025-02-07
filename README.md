# Inital Setup

## Add Sambanova Requirement Key

In a .env file,
`SAMBANOVA_API_KEY=xxxx`.
Pretend this is working on the edge. This will add as the local multimodal model

## Install Required Packages

`pip install -r requirements.txt`

## Start up Qdrant Vector Database

````docker run -p 6333:6333 -p 6334:6334 \
 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
 qdrant/qdrant```
````

# 1. Run Edge App

```
export PYWAGGLE_LOG_DIR=test-run
python3 main.py
```

# 2. Natural Language Search Based on CLIP (Image to Embedding) and Captions (Text to Embedding)

sage_client.ipynb
