from fastapi import FastAPI
from app.api import api_router, root_router

app = FastAPI(title="Speech to Text API")

app.include_router(root_router)

app.include_router(api_router)
