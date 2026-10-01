import json
from pathlib import Path
import pytest
from src.evaluation import evaluate_all,validate_judgments
from src.rag import RAGPipeline,validate_citations,Citation
from src.retrieval import BM25Retriever,DenseRetriever,HybridRetriever,load_manifest
ROOT=Path(__file__).parents[1]
@pytest.fixture
def chunks(): return load_manifest(ROOT/'evaluation/chunk_manifest.json')
def test_manifest_and_judgments_are_valid(chunks):
 d=json.loads((ROOT/'evaluation/retrieval_dataset.json').read_text()); j=json.loads((ROOT/'evaluation/relevance_judgments.json').read_text()); validate_judgments(j,d,chunks)
def test_retrievers_rank_and_reject_empty_query(chunks):
 for cls in (DenseRetriever,BM25Retriever,HybridRetriever):
  r=cls(chunks); assert len(r.search('reflection tokens',3))==3
  with pytest.raises(ValueError): r.search('')
def test_retrieval_metrics_are_bounded():
 for values in evaluate_all().values(): assert all(0<=x<=1 for x in values.values())
def test_grounded_pipeline_uses_retrieved_citations(chunks):
 p=RAGPipeline(chunks,generator=lambda _: 'Grounded fixture.')
 out=p.answer('What are reflection tokens?',2); assert out['answer']=='Grounded fixture.' and len(out['citations'])==2
 with pytest.raises(ValueError): validate_citations([Citation('paper1',1,'missing')],[])
def test_pipeline_input_limits(chunks):
 with pytest.raises(ValueError): RAGPipeline(chunks).answer(' ')
