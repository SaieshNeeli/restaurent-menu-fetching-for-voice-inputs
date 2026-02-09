import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(current_dir)
print(project_root)
from fastapi import FastAPI
from rag_searching import router as rag_router


app = FastAPI(title="Vector Search API")

# Include RAG router with prefix
app.include_router(rag_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
