import json
import time
import uuid
from typing import Dict, List, Optional

from langchain_core.messages import SystemMessage

from config import MODEL, TEMPERATURE
from model.get_model import get_llm
from RAG.evidence import format_evidences_for_prompt, prepare_public_citations
from RAG.evidence_verifier import verify_answer_citations
from RAG.retrieve import retrieve_vector_store


_RESEARCH_JOBS: Dict[str, dict] = {}
_RESEARCH_LLM = None


def start_research_job(question: str, session_id: str) -> dict:
    job_id = str(uuid.uuid4())
    _RESEARCH_JOBS[job_id] = {
        "job_id": job_id,
        "question": question,
        "session_id": session_id,
        "status": "pending",
        "progress": 0,
        "stage": "等待开始",
        "report": "",
        "citations": [],
        "verification": {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    return _RESEARCH_JOBS[job_id]


def get_research_job(job_id: str) -> Optional[dict]:
    return _RESEARCH_JOBS.get(job_id)


def build_research_subquestions(question: str) -> List[str]:
    return [
        f"{question} 的核心法律依据是什么？",
        f"{question} 的适用条件、构成要件或判断标准是什么？",
        f"{question} 当事人可以采取哪些救济路径？",
        f"{question} 有哪些风险、例外和注意事项？",
    ]


def _set_job(job_id: str, **updates):
    job = _RESEARCH_JOBS[job_id]
    job.update(updates)
    job["updated_at"] = time.time()


def _dedupe_evidences(evidences: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for evidence in evidences:
        key = evidence.get("content_hash") or (evidence.get("source_label"), evidence.get("excerpt"))
        if key in seen:
            continue
        seen.add(key)
        copied = dict(evidence)
        copied["citation_id"] = len(unique) + 1
        unique.append(copied)
    return unique


async def retrieve_for_research(query: str, exclude_ids: List[str]) -> str:
    return await retrieve_vector_store.ainvoke({
        "query": query,
        "exclude_ids": exclude_ids,
    })


def get_research_llm():
    global _RESEARCH_LLM
    if _RESEARCH_LLM is None:
        _RESEARCH_LLM = get_llm(MODEL, TEMPERATURE)
    return _RESEARCH_LLM


def _format_sections_for_prompt(sections: List[dict]) -> str:
    if not sections:
        return "未检索到有效研究小结。"
    lines = []
    for index, section in enumerate(sections, start=1):
        lines.extend([
            f"子问题{index}：{section.get('question', '')}",
            f"检索小结：{section.get('text', '')}",
            "",
        ])
    return "\n".join(lines).strip()


async def generate_research_report_with_llm(
    question: str,
    sections: List[dict],
    evidences: List[dict],
) -> str:
    evidence_prompt = format_evidences_for_prompt(evidences) or "当前知识库未检索到可引用证据。"
    section_prompt = _format_sections_for_prompt(sections)
    prompt = f"""你是严谨的中文法律研究助手。请基于系统提供的检索小结和证据，生成一份可验证的深度研究报告。

用户问题：
{question}

检索小结：
{section_prompt}

可引用证据：
{evidence_prompt}

写作要求：
1. 使用 Markdown，结构包含：问题概述、核心结论、法律依据、适用条件/判断要点、风险与例外、行动建议、证据清单。
2. 每个法律判断句必须尽量带上引用编号，例如 [1]、[2]。
3. 只能引用上方存在的证据编号，不能编造法条、案例、文件名或编号。
4. 如果证据不足，要明确写出“当前知识库未检索到直接依据”，并降低结论确定性。
5. 不要大段照抄证据摘录，要用自己的话综合分析。
"""
    response = await get_research_llm().ainvoke([SystemMessage(content=prompt)])
    return getattr(response, "content", str(response)).strip()


def _build_report(question: str, sections: List[dict], evidences: List[dict]) -> str:
    lines = [
        "# 深度研究报告",
        "",
        "## 一、问题概述",
        question,
        "",
        "## 二、核心结论",
    ]
    if sections:
        lines.append("基于当前知识库检索结果，以下结论仅覆盖已检索到直接依据的部分。")
    else:
        lines.append("当前知识库未检索到足够直接依据，无法形成确定结论。")

    lines.extend(["", "## 三、分项研究"])
    for index, section in enumerate(sections, start=1):
        citation = f"[{min(index, len(evidences))}]" if evidences else ""
        lines.extend([
            f"### {index}. {section['question']}",
            f"{section['text']} {citation}".strip(),
            "",
        ])

    lines.extend(["## 四、行动建议"])
    lines.append("建议结合具体事实、合同文本、证据材料和程序期限进一步判断；如涉及重大权益，应咨询专业律师。")

    lines.extend(["", "## 五、证据清单"])
    if not evidences:
        lines.append("未检索到可展示证据。")
    for evidence in evidences:
        lines.append(f"[{evidence['citation_id']}] {evidence['source_label']}：{evidence['excerpt']}")

    return "\n".join(lines)


async def run_research_job(job_id: str) -> None:
    job = get_research_job(job_id)
    if not job:
        return

    try:
        _set_job(job_id, status="running", stage="正在拆解问题", progress=10)
        subquestions = build_research_subquestions(job["question"])
        sections = []
        all_evidences = []
        retrieved_ids = []

        for index, subquestion in enumerate(subquestions, start=1):
            _set_job(
                job_id,
                stage=f"正在检索第 {index}/{len(subquestions)} 个研究问题",
                progress=10 + int(index / len(subquestions) * 55),
            )
            payload = await retrieve_for_research(subquestion, retrieved_ids)
            data = json.loads(payload)
            retrieved_ids.extend(data.get("retrieved_ids", []))
            sections.append({
                "question": subquestion,
                "text": data.get("text", "") or "未检索到直接依据。",
            })
            all_evidences.extend(data.get("evidences", []))

        _set_job(job_id, stage="正在交叉整理证据", progress=75)
        evidences = _dedupe_evidences(all_evidences)
        _set_job(job_id, stage="正在调用LLM生成研究报告", progress=88)
        try:
            report = await generate_research_report_with_llm(job["question"], sections, evidences)
        except Exception as exc:
            report = _build_report(job["question"], sections, evidences)
            report += f"\n\n> LLM研究报告生成失败，已使用模板兜底：{exc}"
        verification = verify_answer_citations(report, evidences)

        _set_job(
            job_id,
            status="completed",
            stage="研究完成",
            progress=100,
            report=report,
            citations=prepare_public_citations(evidences),
            verification=verification,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            stage="研究失败",
            progress=100,
            report=f"深度研究失败：{exc}",
            citations=[],
            verification={"status": "failed", "warnings": [str(exc)]},
        )
