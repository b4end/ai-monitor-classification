from functools import lru_cache
from utils.types import CategoryResult, DataOutput
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

@lru_cache(maxsize=1024)
def get_normalized(text: str) -> set[str]:
    """
    Удаляет знаки препинания, разделяет строку на слова,
    приводит к нижнему регистру и лемматизирует их.
    
    Args:
        text (str): Текст для лемматизации.
    Returns:
        set[str]: Неизменяемое множество лемматизированных слов.
    """
    if not text or not text.strip():
        return set()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    normalized_words = set()

    for token in doc.tokens:
        if token.pos != "PUNCT":
            token.lemmatize(morph_vocab)
            if token.lemma:
                normalized_words.add(token.lemma.lower())

    return normalized_words


def classify(
        message: str, 
        keywords: dict[str, list[str]]
    ) -> DataOutput:
    """
    Классифицирует текст на категории по ключевым-словам.
    
    Args:
        message (str): Исходный текст для классификации.
        keywords (dict): Словарь, где ключ - категория, значение - список ключевых-слов/фраз.
        
    Returns:
        DataOutput: Объект с исходным текстом и списком результатов по категориям.
    """
    if not message or not keywords:
        return DataOutput(message=message if message else "", results=[])

    norm_text = get_normalized(message)
    results = []

    for category_name, keywords_category in keywords.items():
        results_matched = []

        for phrase in keywords_category:
            norm_phrase = get_normalized(phrase)

            if norm_phrase and norm_phrase.issubset(norm_text):
                results_matched.append(phrase)

        score = (
            len(results_matched) / len(keywords_category)
            if len(keywords_category) > 0
            else 0.0
        )

        results.append(
            CategoryResult(
                category=category_name, 
                score=score, 
                matched=results_matched
                )
        )

    results.sort(key=lambda x: x.score, reverse=True)

    return DataOutput(message=message, results=results)
