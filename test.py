from RAG.vector_doc import VectorManager, create_vector_store


if __name__ == "__main__":
    create_vector_store('/home/RAG_agent/files/test_file', re_build = True)

    res = VectorManager.vector_store.search('reararara', search_type = 'similarity')
    print(res)