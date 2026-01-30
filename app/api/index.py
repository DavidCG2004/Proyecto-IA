from fastapi import APIRouter
from fastapi import APIRouter, status
from fastapi.responses import PlainTextResponse
from app.api.train_routes import router as train_router
from app.api.predict_routes import router as predict_router
from app.api.metrics_routes import router as metrics_router
from app.api.config_routes import router as config_router

api_router = APIRouter(prefix="/api")

api_router.include_router(train_router, prefix="/train", tags=["Training"])
api_router.include_router(predict_router, prefix="/predict", tags=["Prediction"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
api_router.include_router(config_router, prefix="/config", tags=["Configuration"])


root_router = APIRouter() 
@root_router.get("/", 
                 summary="Bienvenida a la API",
                 response_class=PlainTextResponse)
async def root():
    return PlainTextResponse(
        content="¡Bienvenido a mi API de Machine Learning! 🚀\n\nEndpoints disponibles:\n• /api/train\n• /api/predict\n• /api/metrics\n• /api/config\n\nVisita /docs para la documentación interactiva.",
        status_code=status.HTTP_200_OK
    )