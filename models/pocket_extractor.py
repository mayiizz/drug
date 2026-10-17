# Backward-compatibility shim — Step 7
# models/pocket_extractor.py → moved to models/protein/pocket_extractor.py
# pipeline/denovo_pipeline.py imports from this path — shim must stay here.
from models.protein.pocket_extractor import PocketExtractor

__all__ = ["PocketExtractor"]
