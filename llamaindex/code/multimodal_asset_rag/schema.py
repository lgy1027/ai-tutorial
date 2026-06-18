from dataclasses import asdict, dataclass, field


@dataclass
class ParsedDocument:
    text: str
    metadata: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SearchHit:
    route: str
    score: float
    asset_id: str
    title: str
    source_type: str
    source_path: str
    evidence: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def key(self) -> str:
        page = self.metadata.get("page", "")
        return f"{self.asset_id}:{page}:{self.route}"

