from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Optional, Dict
import logging
import pickle
from pathlib import Path
import tempfile
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    사용자가 PDF, CSV, 텍스트 등 다양한 형태의 문서를 추가/삭제/수정/검색할 수 있는 벡터DB 관리 클래스입니다.
    FAISS의 IndexIDMap을 사용하여 문서를 효율적으로 관리하고, 디스크에 저장하여 영속성을 보장합니다.
    """
    
    def __init__(self, persistence_path: str, documents: Optional[List[str]] = None):
        """
        VectorStore 객체를 초기화합니다.
        
        Args:
            persistence_path: 인덱스와 문서를 저장할 디렉토리 경로
            documents: 초기 문서 리스트 (선택사항)
        """
        self.persistence_path = Path(persistence_path)
        self.index_file = self.persistence_path / "faiss.index"
        self.data_file = self.persistence_path / "data.pkl"
        
        self.documents: Dict[int, str] = {}
        self.next_id = 0
        
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embedding_dim = 384
        
        # FAISS 인덱스 초기화
        index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIDMap(index)
        
        # 영속성 경로가 없으면 생성
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # 기존 데이터 로드 또는 초기화
        if not self.load_store():
            if documents:
                for doc in documents:
                    self.add_document(doc)


    def save_store(self):
        """인덱스와 문서 데이터를 원자적으로 디스크에 저장합니다."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=self.persistence_path, suffix=".index") as tmp_index_file:
                faiss.write_index(self.index, tmp_index_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, dir=self.persistence_path, suffix=".pkl") as tmp_data_file:
                with open(tmp_data_file.name, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'next_id': self.next_id
                    }, f)
            
            # 원자적 교체
            os.rename(tmp_index_file.name, self.index_file)
            os.rename(tmp_data_file.name, self.data_file)
            
            logger.info(f"VectorStore 저장 완료: {self.persistence_path}")
        except Exception as e:
            logger.error(f"VectorStore 저장 실패: {e}")
            # 임시 파일 정리
            if 'tmp_index_file' in locals() and os.path.exists(tmp_index_file.name):
                os.remove(tmp_index_file.name)
            if 'tmp_data_file' in locals() and os.path.exists(tmp_data_file.name):
                os.remove(tmp_data_file.name)
            raise


    def load_store(self) -> bool:
        """디스크에서 인덱스와 문서 데이터를 로드합니다."""
        if self.index_file.exists() and self.data_file.exists():
            try:
                # FAISS 인덱스 로드
                self.index = faiss.read_index(str(self.index_file))
                logging.info(f"FAISS 인덱스 로드 완료: {self.index.ntotal}, d={self.index.d}")
                
                # 문서 데이터 로드
                with self.data_file.open("rb") as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.next_id = data['next_id']
                logger.info(f"VectorStore 로드 완료: {self.persistence_path}")
                return True
            except Exception as e:
                logger.error(f"VectorStore 로드 실패: {e}")
                return False
        return False


    def add_document(self, doc: str) -> int:
        """
        단일 문서를 추가하고 고유 ID를 반환합니다.
        
        Args:
            doc: 추가할 문서
            
        Returns:
            int: 추가된 문서의 ID
        """
        try:
            doc_id = self.next_id
            embedding = self.embedder.encode([doc], normalize_embeddings=True)
            
            self.index.add_with_ids(np.array(embedding, dtype=np.float32), np.array([doc_id], dtype=np.int64))
            
            self.documents[doc_id] = doc
            self.next_id += 1
            
            self.save_store()
            logger.info(f"문서 추가 완료 (ID: {doc_id})")
            return doc_id
        except Exception as e:
            logger.error(f"문서 추가 실패: {str(e)}")
            raise

    def add_documents(self, docs: List[str]) -> List[int]:
        """
        여러 문서를 한 번에 추가하고 ID 리스트를 반환합니다.

        Args:
            docs: 추가할 문서 리스트

        Returns:
            List[int]: 추가된 문서들의 ID 리스트
        """
        if not docs:
            return []
        
        try:
            doc_ids = [self.next_id + i for i in range(len(docs))]
            embeddings = self.embedder.encode(docs, normalize_embeddings=True)
            
            self.index.add_with_ids(np.array(embeddings, dtype=np.float32), np.array(doc_ids, dtype=np.int64))
            
            for i, doc in enumerate(docs):
                self.documents[doc_ids[i]] = doc
            
            self.next_id += len(docs)
            self.save_store()
            logger.info(f"{len(docs)}개 문서 추가 완료.")
            return doc_ids
        except Exception as e:
            logger.error(f"문서 배치 추가 실패: {str(e)}")
            raise

    def remove_document(self, doc_id: int) -> bool:
        """
        문서 ID에 해당하는 문서를 삭제합니다.
        
        Args:
            doc_id: 삭제할 문서의 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if doc_id not in self.documents:
                logger.warning(f"문서 ID {doc_id}를 찾을 수 없습니다.")
                return False
                
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            del self.documents[doc_id]
            
            self.save_store()
            logger.info(f"문서 삭제 완료 (ID: {doc_id})")
            return True
        except Exception as e:
            logger.error(f"문서 삭제 실패: {str(e)}")
            raise


    def update_document(self, doc_id: int, new_doc: str) -> bool:
        """
        기존 문서를 새로운 내용으로 수정합니다.
        
        Args:
            doc_id: 수정할 문서의 ID
            new_doc: 새로운 문서 내용
            
        Returns:
            bool: 수정 성공 여부
        """
        try:
            if doc_id not in self.documents:
                logger.warning(f"문서 ID {doc_id}를 찾을 수 없습니다.")
                return False
                
            new_embedding = self.embedder.encode([new_doc], normalize_embeddings=True)
            
            self.index.remove_ids(np.array([doc_id], dtype=np.int64))
            self.index.add_with_ids(np.array(new_embedding, dtype=np.float32), np.array([doc_id], dtype=np.int64))
            
            self.documents[doc_id] = new_doc
            
            self.save_store()
            logger.info(f"문서 수정 완료 (ID: {doc_id})")
            return True
        except Exception as e:
            logger.error(f"문서 수정 실패: {str(e)}")
            raise


    def get_document(self, doc_id: int) -> Optional[str]:
        """
        문서 ID에 해당하는 문서를 반환합니다.
        
        Args:
            doc_id: 조회할 문서의 ID
            
        Returns:
            Optional[str]: 문서 내용 또는 None
        """
        return self.documents.get(doc_id)


    def get_documents(self) -> List[str]:
        return list(self.documents.values())


    def search(self, query: str, threshold: float = 0.4, top_k: int = 5) -> List[Dict[str, any]]:
        """
        쿼리와 유사한 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            threshold: 유사도 임계값 (기본값: 0.4)
            top_k: 반환할 최대 문서 수 (기본값: 5)
            
        Returns:
            List[Dict]: 검색 결과 리스트. {'doc_id', 'content', 'similarity'}
        """
        try:
            if self.index.ntotal == 0:
                return []
                
            query_vec = self.embedder.encode([query], normalize_embeddings=True)

            # 안전을 위해 명시적으로 float32로 변환
            query_vec_np = np.array(query_vec, dtype=np.float32)

            k = min(top_k, self.index.ntotal)
            
            distances, ids = self.index.search(query_vec_np, k)

            results = []
            if ids.size > 0:
                for doc_id, sim in zip(ids[0], distances[0]):
                    if doc_id != -1 and sim >= threshold and doc_id in self.documents:
                        results.append({
                            'doc_id': int(doc_id),
                            'content': self.documents[doc_id],
                            'similarity': float(sim)
                        })
            
            logger.info(f"검색 완료: {len(results)}개 결과 (쿼리: {query})")
            return results
        except Exception as e:
            logger.error(f"검색 실패: {str(e)}")
            raise