"""
Dataclass models mirroring DB tables.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class Result:
    id: Optional[int] = None
    image_path: str = ""
    result_path: str = ""
    face_count: int = 0
    kind: str = ""
    detail: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row) -> "Result":
        return cls(
            id=row["id"],
            image_path=row["image_path"],
            result_path=row["result_path"],
            face_count=row["face_count"],
            kind=row["kind"],
            detail=row["detail"] or "",
            created_at=row["created_at"],
        )


@dataclass
class User:
    id: Optional[int] = None
    username: str = ""
    password_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("password_hash", None)
        return d

    @classmethod
    def from_row(cls, row) -> "User":
        return cls(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
        )