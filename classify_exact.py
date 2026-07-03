from utils.types import *
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def get_normalized(text: str) -> set[str]:
    """
    Принимает строку, возвращает множество лемматизированных слов в нижнем регистре.
    Пунктуация удаляется автоматически.
    """
    if not text or not text.strip():
        return set()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    normalized_words = set()

    for token in doc.tokens:
        if token.pos != 'PUNCT':
            token.lemmatize(morph_vocab)
            if token.lemma: 
                normalized_words.add(token.lemma.lower())
            
    return normalized_words

def classify(message: str, stop_words : dict[str, list[str]]) -> DataOutput:
    if not message or not stop_words:
        return DataOutput(message=message if message else "", results=[])

    norm_text = get_normalized(message)
    results = []

    for category_name, stop_words_category in stop_words.items():
        results_matched = []
        
        for phrase in stop_words_category:
            norm_phrase = get_normalized(phrase)

            if norm_phrase and norm_phrase.issubset(norm_text):
                results_matched.append(phrase)
        
        score = len(results_matched) / len(stop_words_category) if len(stop_words_category) > 0 else 0.0


        results.append(CategoryResult(
            category=category_name, 
            score=score,
            matched=results_matched
        ))

    return DataOutput(message=message, results=results)