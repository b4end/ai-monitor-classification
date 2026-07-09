from pydantic import BaseModel


class CategoryResult(BaseModel):
    """
    Стандартный тип данных для хранения информации по пренадлежности
    текста к определенной категории.\n
    Имеет форму:
    ```
    class CategoryResult(
        category: str
        score: float
        matched: list[str]
    )
    ```
    """
    category: str
    score: float
    matched: list[str]


class DataOutput(BaseModel):
    """
    Стандартный тип данных для возврата информации по итогам функции
    классификации для исходного текста по заданным категориям.\n
    Имеет форму:
    ```
    class DataOutput(
        message: str,
        results: list[CategoryResult]
    )
    ```
    """
    message: str
    results: list[CategoryResult]
