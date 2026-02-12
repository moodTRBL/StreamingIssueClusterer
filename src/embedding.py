import logging

from sentence_transformers import SentenceTransformer

from embedding_model import GeneratedEmbedding

        
class Vectorizer:
    def __init__(self) -> None:
        logging.info("모델 로딩중")
        self.dense_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        logging.info("모델 로드 성공")

    def generate(self, ctx: object, title: str, content: str) -> GeneratedEmbedding:
        full_text = title + " " + content
        dense_vec = self.dense_model.encode(full_text, convert_to_numpy=True).tolist()
        return GeneratedEmbedding(dense=dense_vec)
