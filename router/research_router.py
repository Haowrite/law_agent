from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from RAG.research_service import get_research_job, run_research_job, start_research_job


router = APIRouter(prefix="/api/research", tags=["深度研究"])


class ResearchStartRequest(BaseModel):
    question: str
    session_id: str


@router.post("/start")
async def start_research(req: ResearchStartRequest, background_tasks: BackgroundTasks):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="研究问题不能为空")
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="会话不能为空")

    job = start_research_job(req.question.strip(), req.session_id.strip())
    background_tasks.add_task(run_research_job, job["job_id"])
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
    }


@router.get("/status/{job_id}")
async def research_status(job_id: str):
    job = get_research_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="研究任务不存在")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
    }


@router.get("/result/{job_id}")
async def research_result(job_id: str):
    job = get_research_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="研究任务不存在")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "report": job["report"],
        "citations": job["citations"],
        "verification": job["verification"],
    }
