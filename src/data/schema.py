from dataclasses import dataclass
from typing import Optional


@dataclass
class BusinessRecord:
    """Dataclass representing a business record from Source 1, 2, or 3.
    
    Attributes:
        entity_id: Unique record identifier (prefixed with S1-, S2-, or S3-).
        business_name: Name of the business entity.
        business_address: Address of the business.
        country: Open-set country string label (e.g. US, India, France).
        source: Identified source ('source1', 'source2', or 'source3').
    """
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source and self.entity_id:
            prefix = self.entity_id.split("-")[0]
            if prefix == "S1":
                self.source = "source1"
            elif prefix == "S2":
                self.source = "source2"
            elif prefix == "S3":
                self.source = "source3"
            else:
                self.source = "unknown"
