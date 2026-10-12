"""Health and readiness endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def liveness() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {"status": "alive"}


@router.get("/ready", response_model=None)
def readiness(request: Request):
    """Report whether application startup has completed."""
    runtime = getattr(request.app.state, "runtime", None)

    if runtime is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {"application": "not_initialized"},
            },
        )

    return {
        "status": "ready",
        "checks": {"application": "ok"},
    }
