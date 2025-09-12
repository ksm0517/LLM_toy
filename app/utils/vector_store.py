import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    """
    사용자가 PDF, CSV, 텍스트 등 다양한 형태의 문서를 추가/삭제/수정/검색할 수 있는 벡터DB 관리 클래스입니다.
    문서가 추가/수정/삭제될 때마다 임베딩과 벡터DB를 자동으로 갱신합니다.
    """
    def __init__(self, documents=None):
        """DocumentStore 객체를 초기화합니다. documents는 문자열 리스트입니다."""
        self.documents = documents if documents is not None else []
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self._update_index()


    def _update_index(self):
        """문서 임베딩을 생성하고 FAISS 벡터DB를 갱신합니다."""
        if self.documents:
            self.doc_embeddings = self.embedder.encode(self.documents, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
            self.index.add(np.array(self.doc_embeddings))
        else:
            self.doc_embeddings = None
            self.index = None


    def add_document(self, doc: str):
        """문서를 추가하고 벡터DB를 갱신합니다."""
        self.documents.append(doc)
        self._update_index()


    def remove_document(self, idx: int):
        """문서 인덱스(idx)에 해당하는 문서를 삭제하고 벡터DB를 갱신합니다."""
        if 0 <= idx < len(self.documents):
            self.documents.pop(idx)
            self._update_index()


    def update_document(self, idx: int, new_doc: str):
        """문서 인덱스(idx)에 해당하는 문서를 new_doc으로 수정하고 벡터DB를 갱신합니다."""
        if 0 <= idx < len(self.documents):
            self.documents[idx] = new_doc
            self._update_index()


    def get_documents(self):
        """현재 저장된 모든 문서 리스트를 반환합니다."""
        return self.documents


    def search(self, query: str, threshold: float = 0.4):
        """
        쿼리(query)와 임베딩 유사도(threshold) 기준으로 관련 문서를 검색합니다.
        threshold 이상인 문서만 반환합니다.
        """
        if not self.documents or self.index is None:
            return []
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(query_vec), len(self.documents))
        filtered_results = []
        print(f"[VectorDB Search Log] Query: {query}")
        for i, sim in zip(I[0], D[0]):
            print(f"  문서: {self.documents[i]} | 유사도: {sim:.4f} | threshold: {threshold}")
            if sim >= threshold:
                filtered_results.append(self.documents[i])
        return filtered_results