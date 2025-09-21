import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tqdm import tqdm
import re
import time
import os
from DataMining_service.GEO_service.get_region import get_region_dadata


@lru_cache(maxsize=2000)
def cached_get_region(city):
    """Кэширование запросов к API DaData"""
    return get_region_dadata(city)


def detect_gender_by_keywords(text):
    """
    УЛУЧШЕННОЕ определение пола по ключевым словам в тексте
    Возвращает: 'М' (мужской), 'Ж' (женский), '' (не определен)
    """
    if not text or pd.isna(text):
        return ""

    text = str(text).lower()

    # МУЖСКИЕ МАРКЕРЫ (расширенный список)
    male_keywords = [
        # Прямые указания
        'мужчина', 'парень', 'муж', 'отец', 'папа', 'сын', 'брат', 'дедушка', 'мальчик',
        'господин', 'мужского пола', 'джентльмен',

        # КЛЮЧЕВЫЕ МУЖСКИЕ ГЛАГОЛЫ В ПРОШЕДШЕМ ВРЕМЕНИ
        'решил', 'пошел', 'сделал', 'купил', 'взял', 'написал', 'позвонил', 'приехал',
        'остался', 'ушел', 'попросил', 'получил', 'оплатил', 'заплатил', 'открыл', 'закрыл',
        'работал', 'служил', 'учился', 'жил', 'был доволен', 'был недоволен', 'был рад',
        'стал клиентом', 'обратился', 'воспользовался', 'подключился', 'отключился',
        'пригласил', 'разобраться не смог', 'понял', 'узнал', 'выяснил', 'договорился',

        # Причастия и деепричастия мужского рода
        'оформив', 'получив', 'придя', 'купивший', 'взявший', 'решивший',

        # Профессиональные маркеры
        'программист', 'инженер', 'водитель', 'слесарь', 'электрик', 'строитель',
        'военный', 'солдат', 'офицер', 'сержант', 'капитан', 'майор',

        # Мужские имена
        'александр', 'дмитрий', 'максим', 'сергей', 'андрей', 'алексей', 'артем', 'илья',
        'иван', 'роман', 'михаил', 'даниил', 'егор', 'никита', 'матвей', 'тимур',
        'владимир', 'николай', 'антон', 'павел', 'денис', 'юрий', 'станислав',

        # Контекстные фразы
        'служил в армии', 'был в армии', 'отслужил', 'как мужчина', 'будучи мужчиной'
    ]

    # ЖЕНСКИЕ МАРКЕРЫ (расширенный список)
    female_keywords = [
        # Прямые указания
        'женщина', 'девушка', 'жена', 'мать', 'мама', 'дочь', 'сестра', 'бабушка', 'девочка',
        'госпожа', 'женского пола', 'леди', 'дама', 'тетя',

        # КЛЮЧЕВЫЕ ЖЕНСКИЕ ГЛАГОЛЫ В ПРОШЕДШЕМ ВРЕМЕНИ
        'решила', 'пошла', 'сделала', 'купила', 'взяла', 'написала', 'позвонила', 'приехала',
        'осталась', 'ушла', 'попросила', 'получила', 'оплатила', 'заплатила', 'открыла', 'закрыла',
        'работала', 'училась', 'жила', 'была довольна', 'была недовольна', 'была рада',
        'стала клиенткой', 'обратилась', 'воспользовалась', 'подключилась', 'отключилась',
        'родила', 'кормила', 'воспитывала', 'вышла замуж', 'развелась', 'хотела узнать',
        'собираюсь в армению', 'спросила', 'уточнила', 'поняла', 'узнала', 'выяснила',

        # Причастия женского рода
        'оформив', 'получив', 'придя', 'купившая', 'взявшая', 'решившая',

        # Профессиональные маркеры
        'воспитательница', 'учительница', 'медсестра', 'продавщица', 'кассирша',
        'секретарша', 'бухгалтер', 'экономист', 'менеджер', 'консультант',

        # Женские имена
        'анна', 'мария', 'елена', 'наталья', 'ольга', 'татьяна', 'ирина', 'светлана',
        'екатерина', 'людмила', 'галина', 'нина', 'валентина', 'любовь', 'алла',
        'юлия', 'виктория', 'дарья', 'карина', 'алина', 'вера', 'надежда', 'софья',

        # Специфические женские фразы
        'была беременна', 'рожала', 'кормлю грудью', 'в декрете', 'материнство',
        'как мама', 'как женщина'
    ]

    # Счетчики совпадений
    male_score = 0
    female_score = 0

    # УЛУЧШЕННАЯ ПРОВЕРКА мужских маркеров
    for keyword in male_keywords:
        if keyword in text:
            # Даем больший вес ключевым глаголам и прямым указаниям
            if keyword in ['мужчина', 'парень', 'муж', 'отец', 'папа']:
                male_score += 5  # Максимальный приоритет
            elif keyword in ['решил', 'пошел', 'сделал', 'купил', 'взял', 'написал', 'позвонил', 'пригласил']:
                male_score += 3  # Высокий приоритет для четких маркеров
            else:
                male_score += 1

    # УЛУЧШЕННАЯ ПРОВЕРКА женских маркеров
    for keyword in female_keywords:
        if keyword in text:
            # Даем больший вес ключевым глаголам и прямым указаниям
            if keyword in ['женщина', 'девушка', 'жена', 'мать', 'мама']:
                female_score += 5  # Максимальный приоритет
            elif keyword in ['решила', 'пошла', 'сделала', 'купила', 'взяла', 'написала', 'позвонила', 'хотела']:
                female_score += 3  # Высокий приоритет
            else:
                female_score += 1

    # ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ через регулярные выражения

    # Проверяем мужские окончания глаголов
    male_verb_patterns = [
        r'\b\w+ил\b',  # решил, купил, сделал
        r'\b\w+ал\b',  # писал, читал, работал
        r'\b\w+ел\b',  # хотел, умел, смотрел
        r'\b\w+ёл\b',  # пошёл, нашёл
    ]

    for pattern in male_verb_patterns:
        matches = re.findall(pattern, text)
        male_score += len(matches) * 2  # Каждое совпадение +2 балла

    # Проверяем женские окончания глаголов
    female_verb_patterns = [
        r'\b\w+ила\b',  # решила, купила, сделала
        r'\b\w+ала\b',  # писала, читала, работала
        r'\b\w+ела\b',  # хотела, умела, смотрела
        r'\b\w+ёла\b',  # пошла, нашла
    ]

    for pattern in female_verb_patterns:
        matches = re.findall(pattern, text)
        female_score += len(matches) * 2

    # Определяем пол с учетом минимального порога
    if female_score > male_score and female_score >= 2:
        return "Ж"
    elif male_score > female_score and male_score >= 2:
        return "М"
    else:
        return ""


def process_cities_batch(cities_list):
    """Обрабатывает батч городов для получения кодов регионов"""
    return [cached_get_region(city) for city in cities_list]


def save_with_retry(df, base_filename):
    """Пытается сохранить файл с разными именами в случае блокировки"""
    timestamp = int(time.time())
    attempts = [
        f"{base_filename.split('.')[0]}_{timestamp}.csv",
        f"dataset_processed_{timestamp}.csv",
        os.path.join(os.path.expanduser("~"), "Desktop", f"dataset_cleaned_{timestamp}.csv"),
        os.path.join(os.path.expanduser("~"), "Downloads", f"dataset_cleaned_{timestamp}.csv")
    ]

    for filename in attempts:
        try:
            df.to_csv(filename,
                      encoding='windows-1251',
                      index=False,
                      sep=';')
            print(f"✅ Файл сохранен как: {filename}")
            return filename
        except PermissionError:
            print(f"❌ Не удалось сохранить: {filename}")
            continue

    print("❌ Не удалось сохранить файл ни в одну из папок!")
    return None


def main():
    """Основная функция обработки данных"""
    print("🚀 ЗАПУСК СУПЕР-БЫСТРОЙ ОБРАБОТКИ ДАННЫХ")
    print("=" * 50)

    # Загружаем данные
    print("📂 Загружаем данные...")
    try:
        df = pd.read_csv('CompleteDataset.csv',
                         encoding='windows-1251',
                         engine='python',
                         on_bad_lines='skip',
                         sep=None)
        print(f"✅ Загружено {len(df)} строк данных")
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return

    # Очищаем названия городов
    print("🧹 Очищаем названия городов...")
    df['Город'] = df['Город'].str.split('(').str[0].str.strip()

    # Определяем пол по тексту отзыва
    text_columns = ['Текст', 'Отзыв', 'Комментарий', 'Содержание', 'Описание', 'Сообщение']
    text_column = None

    for col in text_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column:
        print(f"👥 Определяем пол по столбцу '{text_column}'...")
        df['Пол'] = df[text_column].apply(detect_gender_by_keywords)

        # Статистика определения пола
        gender_stats = df['Пол'].value_counts()
        print(f"📊 Статистика определения пола:")
        print(f"   Мужской (М): {gender_stats.get('М', 0)}")
        print(f"   Женский (Ж): {gender_stats.get('Ж', 0)}")
        print(f"   Не определен: {gender_stats.get('', 0)}")

        # Показываем примеры
        print("\n🔍 Примеры определения пола:")
        for gender in ['М', 'Ж']:
            sample = df[df['Пол'] == gender].head(2)
            for _, row in sample.iterrows():
                text_preview = str(row[text_column])[:80] + "..."
                print(f"   {gender}: {text_preview}")
    else:
        print("⚠️  Не найден столбец с текстом для определения пола")
        df['Пол'] = ""

    # Обработка городов с получением кодов регионов
    unique_cities = df['Город'].nunique()
    total_cities = len(df)
    print(f"\n🌍 Обработка городов:")
    print(f"   Всего строк: {total_cities}")
    print(f"   Уникальных городов: {unique_cities}")
    print(f"   Экономия за счет кэша: {((total_cities - unique_cities) / total_cities * 100):.1f}%")

    # Многопоточная обработка городов
    batch_size = 50
    cities_list = df['Город'].tolist()
    batches = [cities_list[i:i + batch_size] for i in range(0, len(cities_list), batch_size)]

    print(f"⚡ Обрабатываем {len(batches)} батчей в 8 потоков...")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(process_cities_batch, batches),
            total=len(batches),
            desc="🚀 Турбо-обработка городов",
            unit="батч"
        ))

    # Объединяем результаты
    flattened_results = []
    for batch_result in results:
        flattened_results.extend(batch_result)

    df['Код местности'] = flattened_results

    # Статистика обработки
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Обработка завершена за {elapsed_time:.1f} сек ({elapsed_time / 60:.1f} мин)")
    print(f"🏃 Скорость: {len(df) / elapsed_time:.1f} строк/сек")

    # Статистика кэша
    cache_info = cached_get_region.cache_info()
    print(f"💾 Статистика кэша: попаданий {cache_info.hits}, промахов {cache_info.misses}")

    # Сохраняем результат
    print("\n💾 Сохраняем результат...")
    saved_file = save_with_retry(df, 'CompleteDataset_with_gender_and_regions.csv')

    if saved_file:
        print(f"🎉 УСПЕШНО! Обработано {len(df)} строк")
        print(f"📁 Файл сохранен: {saved_file}")

        # Финальная статистика
        print(f"\n📈 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   📊 Всего записей: {len(df)}")
        print(f"   👥 Определен пол: {len(df[df['Пол'] != ''])}")
        print(f"   🌍 Обработаны города: {len(df[df['Код местности'] != ''])}")
        print(f"   ⚡ Время обработки: {elapsed_time:.1f} сек")
    else:
        print("❌ Ошибка сохранения файла!")


# Запуск программы
if __name__ == "__main__":
    main()
