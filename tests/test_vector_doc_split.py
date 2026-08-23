from langchain_core.documents import Document


def test_split_documents_falls_back_for_markdown_without_articles():
    from RAG.vector_doc import split_documents

    docs = [
        Document(
            page_content="# 中期进展\n\n研究背景和已完成工作。\n\n## 下一步计划\n\n继续实验和论文撰写。",
            metadata={"source": "/tmp/硕士研究生论文中期进展报告-正文内容.md"},
        )
    ]

    chunks = split_documents(docs)

    assert chunks
    assert chunks[0].metadata["article"] == "全文片段1"
    assert chunks[0].metadata["filename"] == "硕士研究生论文中期进展报告-正文内容"


def test_split_documents_supports_arabic_number_articles():
    from RAG.vector_doc import split_documents

    docs = [
        Document(
            page_content="第1条 总则内容。\n\n第2条 其他内容。",
            metadata={"source": "/tmp/示例法规.md"},
        )
    ]

    chunks = split_documents(docs)

    assert [chunk.metadata["article"] for chunk in chunks] == ["第1条", "第2条"]
