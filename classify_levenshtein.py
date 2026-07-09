from functools import lru_cache
from utils.types import CategoryResult, DataOutput
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
import rapidfuzz

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

@lru_cache(maxsize=1024)
def get_normalized(text: str) -> frozenset[str]:
    """
    Удаляет знаки препинания, разделяет строку на слова,
    приводит к нижнему регистру и лемматизирует их.
    
    Args:
        text (str): Текст для лемматизации.
    Returns:
        frozenset[str]: Неизменяемое множество лемматизированных слов.
    """

    if not text or not text.strip():
        return frozenset()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    normalized_words = set()

    for token in doc.tokens:
        if token.pos != "PUNCT":
            token.lemmatize(morph_vocab)
            if token.lemma:
                normalized_words.add(token.lemma.lower())

    return frozenset(normalized_words)

def is_word_in_text_fuzzy(
        target_word: str,  
        min_similarity: float, 
        text_words: frozenset[str]
    ) -> bool:
    """
    Проверяет, есть ли во множестве text_words слово, 
    похожее на target_word с учетом порога min_similarity.
    
    Args:
        target_word (str): Искомое слово.
        min_similarity (float): Число от 0 до 1, порог схожести для совпадения.
        text_words (frozenset[str]): Множество слов, в котором ищем.
        
    Returns:
        bool: True, если найдено похожее слово, иначе False.
    """
    if target_word in text_words:
        return True
    
    cutoff_score = min_similarity * 100.
        
    match = rapidfuzz.process.extractOne(
        target_word, 
        text_words, 
        scorer=rapidfuzz.fuzz.ratio, 
        score_cutoff=cutoff_score
    )
    return match is not None


def classify(
        message: str,
        min_similarity: float, 
        keywords: dict[str, list[str]]
    ) -> DataOutput:
    """
    Классифицирует текст на категории по ключевым-словам.
    
    Args:
        message (str): Исходный текст для классификации.
        min_similarity (float): Порог схожести от 0 до 1.
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

            if not norm_phrase: 
                continue

            is_phrase_matched = all(
                is_word_in_text_fuzzy(word, min_similarity, norm_text) 
                for word in norm_phrase
            )

            if is_phrase_matched:
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
