from pydantic import BaseModel


class CategoryResult(BaseModel):
    category: str
    score: float
    matched: list[str]


class DataOutput(BaseModel):
    message: str
    results: list[CategoryResult]
