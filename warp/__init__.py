# Suppress deprecation warnings from dependencies
import warnings
warnings.filterwarnings("ignore", message=".*openvino.runtime.*deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from .indexer import Indexer
from .searcher import Searcher

from .modeling.checkpoint import Checkpoint
