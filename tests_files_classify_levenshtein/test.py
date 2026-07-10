import json
import time
import sys
import os
import warnings

# Игнорируем предупреждения pymorphy2 (от natasha)
warnings.filterwarnings("ignore", category=UserWarning, module="pymorphy2")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from classify_levenshtein import classify  # noqa: E402


def load_json(filename: str):
    filepath = os.path.join(CURRENT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.decoder.JSONDecodeError:
        print(f"❌ ОШИБКА: Файл '{filename}' пустой или содержит неверный формат JSON!")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл '{filename}' не найден в папке тестов!")
        sys.exit(1)


def run_tests():
    inputs = load_json("test_inputs.json")
    correct_outputs = load_json("test_correct.json")

    if len(inputs) != len(correct_outputs):
        print("❌ ОШИБКА: Количество тестов во входах и выходах не совпадает!")
        return

    actual_outputs = []
    passed_count = 0

    print("Запуск автотестов нечеткого поиска (Levenshtein)...\n")

    start_total_time = time.perf_counter()

    for i, (test_input, expected_output) in enumerate(zip(inputs, correct_outputs)):
        message = test_input.get("message")
        min_similarity = test_input.get("min_similarity", 0.8)
        keywords = test_input.get("keywords")

        result_obj = classify(
            message=message, 
            min_similarity=min_similarity, 
            keywords=keywords
        )

        try:
            result_dict = result_obj.model_dump()
        except AttributeError:
            result_dict = result_obj.dict()

        actual_outputs.append(result_dict)

        if result_dict == expected_output:
            passed_count += 1
        else:
            print(f"⚠️ Тест {i+1} не совпал!")
            print(f"  Описание:  {test_input.get('description', '')}")
            print(f"  Ожидалось: {expected_output}")
            print(f"  Получено:  {result_dict}\n")

    end_total_time = time.perf_counter()
    total_execution_time = end_total_time - start_total_time

    # Записываем фактические результаты, чтобы их можно было легко сравнить в случае ошибки
    output_filepath = os.path.join(CURRENT_DIR, "test_outputs.json")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(actual_outputs, f, ensure_ascii=False, indent=2)

    total_count = len(inputs)
    print("--- ИТОГИ ---")
    if passed_count == total_count:
        print(f"✅ УСПЕШНО ({passed_count}/{total_count})")
    else:
        print(f"❌ ПРОВАЛ ({passed_count}/{total_count})")

    print(f"Время выполнения всех тестов: {total_execution_time:.4f} сек.")


if __name__ == "__main__":
    run_tests()