import logging
from typing import Any, Dict, Union

from src.pipeline.train import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train(config: Union[str, Dict[str, Any]] = "configs/config.yaml") -> Dict[str, Any]:
    """Top-level training entry point.
    
    Args:
        config: Path to YAML config file or config dictionary.
        
    Returns:
        Dictionary containing evaluation metrics and pipeline execution statistics.
    """
    logger.info("Starting training pipeline...")
    return run_training_pipeline(config=config)


if __name__ == "__main__":
    train()
