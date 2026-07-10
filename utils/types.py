from pydantic import BaseModel


class CategoryResult(BaseModel):
    """
    Хранит информацию по пренадлежности текста к определенной категории.

    Args:
        category (str): Название категории
        score (float): Число от 0 до 1, порог схожести для совпадения.
        matched (list[str]): Совпавшие ключевые слова
    """
    category: str
    score: float
    matched: list[str]


class DataOutput(BaseModel):
    """
    Хранит итоги классификации для текста по заданным категориям.
    
    Args:
        message (str): Текст для классификации 
        results (list[CategoryResult]): Результаты по картегориям
    """
    message: str
    results: list[CategoryResult]
