import logging
from typing import Any, Dict, Union

from src.inference.predict import run_inference_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_inference(config: Union[str, Dict[str, Any]] = "configs/config.yaml") -> None:
    """Top-level test dataset inference entry point.
    
    Args:
        config: Path to YAML config file or config dictionary.
    """
    logger.info("Delegating to inference pipeline...")
    return run_inference_pipeline(config=config)


if __name__ == "__main__":
    run_inference()
