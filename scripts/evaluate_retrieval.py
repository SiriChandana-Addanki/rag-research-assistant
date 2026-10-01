import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
from src.evaluation import evaluate_all
if __name__ == '__main__':
 result=evaluate_all(); out=Path('evaluation/retrieval_results.json'); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
