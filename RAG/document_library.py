import os
import re
import hmac
import hashlib
from pathlib import Path


SUPPORTED_DOCUMENT_EXTENSIONS = {".md", ".txt", ".docx"}


def safe_document_filename(filename: str) -> str:
    raw_name = Path(filename or "").name
    name = raw_name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-\.\u4e00-\u9fff]", "_", name)
    suffix = Path(name).suffix.lower()

    if not name or name in {".", ".."}:
        raise ValueError("文件名不能为空")
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {suffix}")

    return name


def document_key_for_path(doc_path: str, secret: str) -> str:
    normalized_path = str(Path(doc_path).expanduser().resolve())
    digest = hmac.new(
        (secret or "rag-agent-document-key").encode("utf-8"),
        normalized_path.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest[:32]


def document_library_root(file_path: str) -> Path:
    root = Path(file_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_document_path(file_path: str, doc_path: str) -> Path:
    root = document_library_root(file_path)
    target = Path(doc_path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("文档路径必须位于文档库目录内") from exc
    return target


def read_document_preview(path: Path, max_chars: int = 4000) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {suffix}")
    if suffix == ".docx":
        return "DOCX 文档已入库，当前仅支持在预览中显示文本类文档内容。"

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read(max_chars)


def remove_file_if_exists(path: Path) -> bool:
    if path.exists() and path.is_file():
        os.remove(path)
        return True
    return False
