import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter, defaultdict
import umap
import hdbscan

warnings.filterwarnings('ignore')


# Функция для красивого вывода прогресса
def print_step(message):
    print(f"\n{'=' * 60}")
    print(f"🚀 {message}")
    print(f"{'=' * 60}")


# Добавляем функцию поиска оптимальных параметров UMAP
def find_optimal_umap_params(embeddings, max_components=50, n_trials=10):
    """Поиск оптимальных параметров для UMAP"""
    print("🔍 Подбираем оптимальные параметры для UMAP...")

    best_score = -1
    best_params = {}
    best_embeddings = None

    # Параметры для перебора
    n_components_options = [20, 30, 40, 50, 60, 70, 80,90]
    n_neighbors_options = [10, 15, 20, 25, 30]
    min_dist_options = [0.1, 0.2, 0.3]

    # Случайный поиск для экономии времени
    trials = min(n_trials, len(n_components_options) * len(n_neighbors_options) * len(min_dist_options))

    for trial in tqdm(range(trials), desc="Поиск параметров UMAP"):
        # Случайно выбираем параметры
        n_components = np.random.choice(n_components_options)
        n_neighbors = np.random.choice(n_neighbors_options)
        min_dist = np.random.choice(min_dist_options)

        try:
            # Применяем UMAP
            umap_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                metric='cosine',
                low_memory=False
            )

            embeddings_reduced = umap_reducer.fit_transform(embeddings)

            # Нормализуем
            minmax_scaler = MinMaxScaler()
            embeddings_normalized = minmax_scaler.fit_transform(embeddings_reduced)

            # Быстрая оценка с помощью мини-кластеринга
            test_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=30,
                min_samples=5,
                cluster_selection_epsilon=0.1,
                metric='euclidean'
            )

            test_labels = test_clusterer.fit_predict(embeddings_normalized)

            # Оцениваем качество
            non_noise_mask = test_labels != -1
            if sum(non_noise_mask) > 10:  # Достаточно точек для оценки
                score = silhouette_score(embeddings_normalized[non_noise_mask],
                                         test_labels[non_noise_mask])

                # Учитываем также процент шума
                noise_percentage = (list(test_labels).count(-1) / len(test_labels)) * 100
                adjusted_score = score * (1 - noise_percentage / 200)  # Штрафуем за высокий шум

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_params = {
                        'n_components': n_components,
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'silhouette_score': score,
                        'noise_percentage': noise_percentage,
                        'adjusted_score': adjusted_score
                    }
                    best_embeddings = embeddings_normalized.copy()

        except Exception as e:
            continue

    print(f"✅ Наилучшие параметры UMAP: {best_params}")
    return best_params, best_embeddings

# Улучшенная функция очистки текста с сохранением банковской терминологии
def improved_clean_text(text):
    """Улучшенная очистка текста с сохранением финансовых терминов"""
    text = str(text).lower()

    # Удаляем только действительно мешающие элементы
    patterns = [
        r'http\S+',  # URL
        r'\S+@\S+',  # email
        r'\+7\s?\(?\d{3}\)?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}',  # телефоны
        r'\b\d{1,2}\s?(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s?\d{4}?\b',
        # даты
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text)

    # Сохраняем цифры и финансовые символы
    text = re.sub(r'[^\w\s\d%$€₽]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def is_valid_review(text):
    """Проверяет, является ли отзыв осмысленным текстом"""
    words = text.split()
    # Отсеиваем слишком короткие отзывы
    if len(words) < 5:
        return False
    # Отсеиваем отзывы, состоящие из одного повторяющегося слова/фразы
    word_count = Counter(words)
    most_common_count = word_count.most_common(1)[0][1]
    if most_common_count / len(words) > 0.5:  # Если одно слово занимает больше 50% текста
        return False
    return True


def find_optimal_hdbscan_params(embeddings):
    """Поиск оптимальных параметров для HDBSCAN"""
    print("🔍 Подбираем оптимальные параметры для HDBSCAN...")
    best_score = -1
    best_params = {}

    # Перебираем различные комбинации параметров
    param_combinations = [
        {'min_cluster_size': 20, 'min_samples': 3, 'epsilon': 0.05},
        {'min_cluster_size': 20, 'min_samples': 5, 'epsilon': 0.1},
        {'min_cluster_size': 20, 'min_samples': 10, 'epsilon': 0.15},
        {'min_cluster_size': 30, 'min_samples': 3, 'epsilon': 0.1},
        {'min_cluster_size': 30, 'min_samples': 5, 'epsilon': 0.15},
        {'min_cluster_size': 30, 'min_samples': 10, 'epsilon': 0.05},
        {'min_cluster_size': 40, 'min_samples': 3, 'epsilon': 0.15},
        {'min_cluster_size': 40, 'min_samples': 5, 'epsilon': 0.05},
        {'min_cluster_size': 40, 'min_samples': 10, 'epsilon': 0.1},
    ]

    for params in tqdm(param_combinations, desc="Поиск оптимальных параметров"):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            cluster_selection_epsilon=params['epsilon'],
            metric='euclidean'
        )

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Оцениваем только если есть кластеры
        if n_clusters > 1:
            non_noise_mask = labels != -1
            if sum(non_noise_mask) > 1:
                try:
                    score = silhouette_score(embeddings[non_noise_mask],
                                             labels[non_noise_mask])
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_params['score'] = score
                        best_params['n_clusters'] = n_clusters
                        best_params['noise_percentage'] = (list(labels).count(-1) / len(labels)) * 100
                except:
                    continue

    return best_params


def refine_clusters(labels, min_size=50):
    """Дополнительная очистка и объединение мелких кластеров"""
    # Объединяем очень мелкие кластеры
    label_counts = Counter(labels)
    small_clusters = [label for label, count in label_counts.items()
                      if count < min_size and label != -1]

    # Переназначаем мелкие кластеры как шум
    refined_labels = labels.copy()
    for label in small_clusters:
        refined_labels[refined_labels == label] = -1

    return refined_labels


# Шаг 1: Загрузка данных
print_step("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
print("📂 Загружаем данные из CSV файла...")

try:
    df = pd.read_csv('GaspromBank_dataset.csv', sep=';', encoding='windows-1251', on_bad_lines='warn')
    reviews = df['Отзыв'].dropna().tolist()
    print(f"✅ Успешно загружено {len(reviews)} отзывов")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit()

# Шаг 2: Улучшенная очистка текста и фильтрация
print_step("ШАГ 2: УЛУЧШЕННАЯ ОЧИСТКА ТЕКСТА И ФИЛЬТРАЦИЯ")
print("🧹 Тщательно очищаем текст и фильтруем мусорные отзывы...")

cleaned_reviews = []
original_reviews_filtered = []

for review in tqdm(reviews, desc="Очистка и фильтрация"):
    cleaned = improved_clean_text(review)
    if is_valid_review(cleaned):
        cleaned_reviews.append(cleaned)
        original_reviews_filtered.append(review)

print(f"✅ Очищено {len(cleaned_reviews)} отзывов (отфильтровано {len(reviews) - len(cleaned_reviews)} мусорных)")

# Шаг 3: Создание семантических эмбеддингов
print_step("ШАГ 3: СОЗДАНИЕ СЕМАНТИЧЕСКИХ ЭМБЕДДИНГОВ")
print("🤖 Создаем улучшенные эмбеддинги...")

try:
    # Используем мощную многоязычную модель
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2') #768-dim
    #model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') #384-dim
    print("✅ Улучшенная модель загружена")

    print("📊 Создаем эмбеддинги...")
    embeddings = model.encode(cleaned_reviews,
                              show_progress_bar=True,
                              batch_size=32,
                              convert_to_numpy=True)
    print(f"✅ Создано {len(embeddings)} эмбеддингов размерностью {embeddings.shape[1]}")

except Exception as e:
    print(f"❌ Ошибка при создании эмбеддингов: {e}")
    exit()

# Заменяем шаг 4 на оптимизированную версию
print_step("ШАГ 4: ОПТИМИЗАЦИЯ UMAP И СНИЖЕНИЕ РАЗМЕРНОСТИ")
print("🎯 Подбираем оптимальные параметры UMAP...")

# Поиск оптимальных параметров UMAP
optimal_umap_params, embeddings_normalized = find_optimal_umap_params(
    embeddings,
    max_components=50,
    n_trials=15
)

# Шаг 5: Оптимизация параметров и кластеризация с помощью HDBSCAN
print_step("ШАГ 5: ОПТИМИЗАЦИЯ ПАРАМЕТРОВ И КЛАСТЕРИЗАЦИЯ")

# Используем оптимальные параметры UMAP для финального преобразования
print("🔄 Применяем оптимальные параметры UMAP для финального преобразования...")

final_umap_reducer = umap.UMAP(
    n_components=optimal_umap_params['n_components'],
    n_neighbors=optimal_umap_params['n_neighbors'],
    min_dist=optimal_umap_params['min_dist'],
    random_state=42,
    metric='cosine',
    low_memory=False
)

embeddings_final = final_umap_reducer.fit_transform(embeddings)

# Нормализуем
minmax_scaler = MinMaxScaler()
embeddings_normalized = minmax_scaler.fit_transform(embeddings_final)

print(f"✅ Финальная размерность: {embeddings_normalized.shape}")

# Поиск оптимальных параметров
optimal_params = find_optimal_hdbscan_params(embeddings_normalized)
print(f"✅ Наилучшие параметры: {optimal_params}")

# Кластеризация с оптимальными параметрами
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=optimal_params['min_cluster_size'],
    min_samples=optimal_params['min_samples'],
    cluster_selection_epsilon=optimal_params['epsilon'],
    metric='euclidean',
    cluster_selection_method='leaf',
    prediction_data=True
)

labels = clusterer.fit_predict(embeddings_normalized)

# Дополнительная очистка мелких кластеров
labels = refine_clusters(labels, min_size=100)

# Шаг 5.1: Многоуровневая обработка шума
print("🔄 Запускаем многоуровневую обработку шума...")

# 1. Первый уровень: DBSCAN с разными параметрами
noise_mask = labels == -1
noise_embeddings = embeddings_normalized[noise_mask]

if len(noise_embeddings) > 1000:
    # Пробуем разные параметры для выделения кластеров в шуме
    for eps in [0.3, 0.4, 0.5]:
        secondary_clusterer = DBSCAN(eps=eps, min_samples=15)
        noise_labels_secondary = secondary_clusterer.fit_predict(noise_embeddings)

        # Находим новые кластеры (не шум)
        new_clusters_mask = noise_labels_secondary != -1
        if sum(new_clusters_mask) > 200:  # Если найдены значимые кластеры
            break

    # Переназначаем метки
    max_original_label = max(labels)
    new_labels_for_noise = np.where(noise_labels_secondary != -1,
                                    noise_labels_secondary + max_original_label + 1,
                                    -1)

    labels[noise_mask] = new_labels_for_noise
    print(f"✅ Выделено {len(set(new_labels_for_noise)) - 1} дополнительных кластеров из шума")

# Обновляем статистику
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(labels).count(-1)

print(f"✅ Найдено {n_clusters} кластеров")
print(f"📊 Выявлено {n_noise} шумовых точек ({n_noise / len(cleaned_reviews) * 100:.1f}%)")

# Шаг 6: Продвинутый анализ кластеров
print_step("ШАГ 6: ПРОДВИНУТЫЙ АНАЛИЗ КЛАСТЕРОВ")
print("🔍 Глубокий анализ содержания кластеров...")

# Создаем DataFrame с результатами
results_df = pd.DataFrame({
    'review': cleaned_reviews,
    'original_review': original_reviews_filtered,
    'cluster': labels
})

# Расширенные русские стоп-слова с банковской терминологией
extended_russian_stopwords = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
    'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
    'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
    'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до',
    'вас', 'нибудь', 'опять', 'уж', 'вам', 'сказал', 'ведь', 'там', 'потом', 'себя', 'ничего',
    'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их',
    'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'ж',
    'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь',
    'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'кажется', 'сейчас', 'были', 'куда',
    'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
    'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
    'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед',
    'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно',
    'всю', 'между', 'это', 'вот', 'там', 'тут', 'здесь', 'очень', 'просто', 'сильно', 'свой',
    # Банковские стоп-слова
    'банк', 'банка', 'банке', 'банку', 'банком', 'банки', 'банков',
    'карта', 'карты', 'карту', 'карте', 'картой', 'карт',
    'газпромбанк', 'газпромбанка', 'газпромбанке', 'газпромбанку', 'газпромбанком',
    'счёт', 'счета', 'счёте', 'счету', 'счетом', 'счетов',
    'деньги', 'деньгам', 'деньгами', 'денег',
    'рубль', 'рублей', 'рублям', 'рублями', 'руб', 'р',
    'отделение', 'отделения', 'отделению', 'отделением', 'отделений',
    'клиент', 'клиента', 'клиенту', 'клиентом', 'клиенты', 'клиентов',
    'услуга', 'услуги', 'услугу', 'услугой', 'услуг',
    'приложение', 'приложения', 'приложению', 'приложением', 'приложений',
    'мобильный', 'мобильного', 'мобильному', 'мобильным', 'мобильные',
    'онлайн', 'интернет', 'веб'
]


# Улучшенная функция для получения топовых слов
def get_enhanced_top_words(cluster_texts, n_words=15):
    """Получаем улучшенные топовые слова с учетом n-grams и исключением стоп-слов"""
    # Создаем CountVectorizer для анализа n-grams
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=extended_russian_stopwords,
        min_df=2,
        max_df=0.8
    )

    try:
        X = count_vectorizer.fit_transform(cluster_texts)
        feature_names = count_vectorizer.get_feature_names_out()

        # Суммируем частоты
        sums = X.sum(axis=0)
        top_indices = np.argsort(sums.A1)[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        return top_words
    except:
        return ["недостаточно_данных"]


# Расширенный словарь категорий с регулярными выражениями
category_patterns = {
    'Кредитные карты': [
        r'кредитн[а-я]* карт[а-я]*', r'кредитк[а-я]*', r'кэшбэк',
        r'льготн[а-я]* период', r'годов[а-я]* обслуживан', r'кредитн[а-я]* лимит',
        r'беспроцентн[а-я]* период', r'платежн[а-я]* систем[а-я]*',
        r'процентн[а-я]* ставк[а-я]*', r'бонусн[а-я]* программ[а-я]*',
        r'миль[и]*', r'балл[ыов]*'
    ],
    'Дебетовые карты': [
        r'дебетов[а-я]* карт[а-я]*', r'дебетк[а-я]*', r'выпуск карт[а-я]*',
        r'доставк[а-я]* карт[а-я]*', r'снят[иье]* наличн[а-я]*',
        r'зарплатн[а-я]* карт[а-я]*', r'пенсионн[а-я]* карт[а-я]*',
        r'премиальн[а-я]* карт[а-я]*', r'моментальн[а-я]* карт[а-я]*'
    ],
    'Мобильное приложение': [
        r'мобильн[а-я]* приложен', r'онлайн банк', r'личн[а-я]* кабинет',
        r'вход в приложен', r'функционал приложен', r'обновлен приложен',
        r'интернет банк', r'уведомлен[ия]*', r'пуш уведомлен',
        r'биометр[ия]*', r'qr[\s-]*код'
    ],
    'Дистанционное обслуживание': [
        r'дистанционн[а-я]* обслуживан', r'удаленн[а-я]* служба',
        r'онлайн[\s-]*услуг[и]*', r'видеоконсультац[ия]*',
        r'электронн[а-я]* подпис[ь]*', r'удаленн[а-я]* идентификац[ия]*',
        r'цифров[а-я]* служба'
    ],
    'Ипотека': [
        r'ипотек[а-я]*', r'ипотечн[а-я]* кредит', r'первоначальн[а-я]* взнос',
        r'ипотечн[а-я]* ставк[а-я]*', r'недвижимость', r'жилищн[а-я]* кредит',
        r'ипотечн[а-я]* договор', r'квартир[аы]* в ипотек',
        r'рефинансирован[ие]* ипотек', r'льготн[а-я]* ипотек'
    ],
    'Автокредит': [
        r'автокредит[а-я]*', r'авто[\s-]*кредит', r'автомобил[ья]* в кредит',
        r'авто[\s-]*займ', r'покупк[аи]* автомобил[я]*',
        r'авто[\s-]*салон', r'транспортн[а-я]* средств[ао]*',
        r'льготн[а-я]* автокредит'
    ],
    'Потребительские кредиты': [
        r'кредит наличн[а-я]*', r'заявк[а-я]* на кредит', r'одобрен[ие]* кредит',
        r'погашен[ие]* кредит', r'кредитн[а-я]* истори[яи]',
        r'процентн[а-я]* ставк[а-я]*', r'кредитн[а-я]* договор',
        r'потребительск[а-я]* кредит', r'нецелев[а-я]* кредит'
    ],
    'Вклады и счета': [
        r'вклад[а-я]*', r'депозит[а-я]*', r'процентн[а-я]* ставк[а-я]*',
        r'накопительн[а-я]* счет', r'сберегательн[а-я]* счет',
        r'пополнен[ие]* вклад', r'срочн[а-я]* вклад', r'проценты по вклад',
        r'расчетн[а-я]* счет', r'текущ[а-я]* счет'
    ],
    'Долгосрочные сбережения': [
        r'долгосрочн[а-я]* сбережен', r'накоплен[ие]* на будущее',
        r'целев[а-я]* накоплен', r'сбережен[ия]* дет[яей]*',
        r'образовательн[а-я]* накоплен', r'сберегательн[а-я]* программ[а-я]*',
        r'накопительн[а-я]* программ[а-я]*'
    ],
    'Пенсионные продукты': [
        r'пенсионн[а-я]* накоплен', r'негосударственн[а-я]* пенсионн[а-я]*',
        r'нпф', r'пенсионн[а-я]* программ[а-я]*',
        r'добровольн[а-я]* пенсионн[а-я]*', r'пенсионн[а-я]* счет',
        r'накоплен[ие]* на пенсию', r'пенсионн[а-я]* обеспечен'
    ],
    'Инвестиции': [
        r'инвестиц[ия]*', r'брокерск[а-я]* счет', r'ценн[а-я]* бумаг[и]*',
        r'фондов[а-я]* рынок', r'акци[ия]*', r'облигац[ия]*',
        r'па[её]* фонд', r'инвестиционн[а-я]* программ[а-я]*',
        r'управлен[ие]* активами', r'трейдинг'
    ],
    'Металлические счета': [
        r'металлич[а-я]* счет', r'золот[а-я]* счет',
        r'обезличенн[а-я]* металлич[а-я]* счет', r'омс',
        r'покупк[аи]* золот[а-я]*', r'драгоценн[а-я]* металл[ыов]*',
        r'серебр[оа]*', r'платин[аы]*'
    ],
    'Страхование': [
        r'страхов[а-я]*', r'страхов[а-я]* полис', r'страхов[а-я]* компания',
        r'страхов[а-я]* выплат[аы]*', r'страхов[а-я]* случа[йя]*',
        r'страхов[а-я]* премия', r'добровольн[а-я]* страхован',
        r'обязательн[а-я]* страхован', r'каско', r'осаго'
    ],
    'Платежи и переводы': [
        r'платеж[а-я]*', r'перевод[а-я]*', r'денежн[а-я]* перевод',
        r'оплат[а-я]* услуг', r'коммунальн[а-я]* платеж',
        r'квитанц[ия]*', r'автоплатеж', r'межбанковск[а-я]* перевод',
        r'быстр[а-я]* перевод', r'перевод на карт'
    ],
    'Экосистема и партнеры': [
        r'экосистем[а-я]*', r'партнерск[а-я]* программ[а-я]*',
        r'бонусн[а-я]* систем[а-я]*', r'лояльност[ьи]*',
        r'скидк[аи]* партнер', r'совместн[а-я]* акция',
        r'кобрендинг[а-я]*', r'совместн[а-я]* карт[а-я]*'
    ]
}


def advanced_category_detection(cluster_texts, category_patterns, top_words_list):
    """Продвинутое определение категории на основе анализа контекста и ключевых слов"""
    if not cluster_texts:
        return 'Неопределенная'

    all_text = ' '.join(cluster_texts).lower()
    word_count = len(all_text.split())

    category_scores = {}

    # 1. Подсчет очков по регулярным выражениям
    for category, patterns in category_patterns.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, all_text)
            score += len(matches) * 3
        if word_count > 0:
            normalized_score = (score / word_count) * 1000
        else:
            normalized_score = 0
        category_scores[category] = normalized_score

    # 2. Критическое улучшение: Анализ топовых слов
    top_words_str = ' '.join(top_words_list).lower()
    top_words_score = {}
    for category, patterns in category_patterns.items():
        score = 0
        for pattern in patterns:
            # Ищем паттерны не во всем тексте, а в топовых словах
            matches = re.findall(pattern, top_words_str)
            score += len(matches) * 5  # Даем больший вес совпадениям в топовых словах
        top_words_score[category] = score

    # 3. Объединяем оценки (50% - регулярки, 50% - топовые слова)
    combined_scores = {}
    for cat in category_scores.keys():
        combined_scores[cat] = (category_scores.get(cat, 0) + top_words_score.get(cat, 0) * 10)  # Масштабируем

    # 4. Определяем победителя с более высоким порогом уверенности
    if combined_scores:
        best_category, best_score = max(combined_scores.items(), key=lambda x: x[1])
        second_best_score = sorted(combined_scores.values())[-2] if len(combined_scores) > 1 else 0

        # Если победитель не сильно оторвался от второго места, или общий счет низкий - помечаем как обслуживание
        if best_score >= 15 and (best_score - second_best_score) > (best_score * 0.3):
            return best_category
        else:
            return 'Обслуживание'
    return 'Обслуживание'


# Анализируем каждый кластер (исключая шум -1)
cluster_info = []
valid_cluster_ids = [label for label in unique_labels if label != -1]

for cluster_id in tqdm(valid_cluster_ids, desc="Анализ кластеров"):
    cluster_mask = results_df['cluster'] == cluster_id
    cluster_reviews = results_df[cluster_mask]['review'].tolist()
    cluster_size = len(cluster_reviews)

    if cluster_size > 0:
        top_words = get_enhanced_top_words(cluster_reviews, n_words=12)

        # Определяем категорию с учетом топовых слов
        category = advanced_category_detection(cluster_reviews, category_patterns, top_words)

        # Берем репрезентативные примеры
        sample_reviews = cluster_reviews[:2] if len(cluster_reviews) >= 2 else cluster_reviews

        cluster_info.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'top_words': ', '.join(top_words),
            'category': category,
            'sample_reviews': sample_reviews
        })

# Сортируем по размеру кластера
cluster_info.sort(key=lambda x: x['size'], reverse=True)

# Шаг 6.1: Объединение дублирующихся категорий
print("🔄 Объединяем дублирующиеся категории...")

# Создаем словарь для объединения
merge_map = {
    'Кредитные карты': ['Кредитные карты'],
    'Дебетовые карты': ['Дебетовые карты'],
    'Дистанционное обслуживание': ['Дистанционное обслуживание'],
    'Страхование': ['Страхование'],
    'Ипотека': ['Ипотека'],
    'Автокредит': ['Автокредит'],
    'Мобильное приложение': ['Мобильное приложение'],
    'Потребительские кредиты': ['Потребительские кредиты'],
    'Вклады и счета': ['Вклады и счета'],
    'Инвестиции': ['Инвестиции'],
    'Долгосрочные сбережения': ['Долгосрочные сбережения'],
    'Пенсионные продукты': ['Пенсионные продукты'],
    'Металлические счета': ['Металлические счета'],
    'Экосистема и партнеры': ['Экосистема и партнеры'],
    'Платежи и переводы': ['Платежи и переводы']
}

# Перебираем кластеры и перезаписываем категории согласно merge_map
for info in cluster_info:
    current_cat = info['category']
    for new_cat, old_cats in merge_map.items():
        if current_cat in old_cats:
            info['category'] = new_cat
            break

# Теперь группируем кластеры по новой категории и объединяем их
merged_cluster_info = []
category_groups = defaultdict(list)

for info in cluster_info:
    category_groups[info['category']].append(info)

# Создаем новый список кластеров, объединяя те, что в одной категории
for category, clusters in category_groups.items():
    if len(clusters) == 1:
        # Если кластер один в категории, просто добавляем его
        merged_cluster_info.append(clusters[0])
    else:
        # Если несколько кластеров в одной категории, объединяем их
        combined_size = sum(cluster['size'] for cluster in clusters)
        # Объединяем топ-слова и выбираем уникальные
        all_top_words = []
        for cluster in clusters:
            all_top_words.extend(cluster['top_words'].split(', '))
        # Берем самые частые уникальные слова из объединенного списка
        top_words_combined = [word for word, count in Counter(all_top_words).most_common(12)]
        # Берем примеры отзывов из самого большого кластера
        sample_reviews = max(clusters, key=lambda x: x['size'])['sample_reviews']

        merged_cluster_info.append({
            'cluster_id': f"Merged_{category}",
            'size': combined_size,
            'top_words': ', '.join(top_words_combined),
            'category': category,
            'sample_reviews': sample_reviews
        })

# Заменяем старый cluster_info на объединенный
cluster_info = merged_cluster_info
# Сортируем по размеру
cluster_info.sort(key=lambda x: x['size'], reverse=True)

print("\n📊 РАСПРЕДЕЛЕНИЕ КЛАСТЕРОВ:")
for info in cluster_info:
    print(f"\n🔸 Кластер {info['cluster_id']} ({info['size']} отзывов) - {info['category']}:")
    print(f"   Ключевые слова: {info['top_words']}")
    if info['sample_reviews']:
        print(f"   Пример: '{info['sample_reviews'][0][:100]}...'")

# Шаг 7: Сохранение результатов
print_step("ШАГ 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("💾 Сохраняем все результаты...")

# Добавляем категории в основной DataFrame
category_map = {info['cluster_id']: info['category'] for info in cluster_info}
results_df['category'] = results_df['cluster'].map(category_map)
# Для шумовых точек ставим категорию "Шум"
results_df.loc[results_df['cluster'] == -1, 'category'] = 'Шум'

# Сохраняем результаты
results_df.to_csv('banking_reviews_clustered_improved.csv', index=False, encoding='utf-8')
print("✅ Результаты сохранены: banking_reviews_clustered_improved.csv")

# Сохраняем детальную информацию о кластерах
cluster_details = []
for info in cluster_info:
    cluster_details.append({
        'cluster_id': info['cluster_id'],
        'size': info['size'],
        'category': info['category'],
        'top_words': info['top_words'],
        'sample_review_1': info['sample_reviews'][0] if len(info['sample_reviews']) > 0 else '',
        'sample_review_2': info['sample_reviews'][1] if len(info['sample_reviews']) > 1 else ''
    })

cluster_df = pd.DataFrame(cluster_details)
cluster_df.to_csv('clusters_info_detailed.csv', index=False, encoding='utf-8')
print("✅ Детальная информация о кластерах сохранена: clusters_info_detailed.csv")

# Шаг 8: Визуализация
print_step("ШАГ 8: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("📊 Создаем визуализации...")

# Визуализация распределения категорий
plt.figure(figsize=(14, 10))
category_sizes = {info['category']: info['size'] for info in cluster_info}
category_df = pd.DataFrame(list(category_sizes.items()), columns=['Category', 'Size'])
category_df = category_df.sort_values('Size', ascending=True)

plt.barh(category_df['Category'], category_df['Size'], color=plt.cm.Set3(np.arange(len(category_df))))
plt.title('Распределение отзывов по категориям', fontsize=14, fontweight='bold')
plt.xlabel('Количество отзывов', fontsize=12)
plt.tight_layout()
plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
print("✅ График распределения категорий сохранен: category_distribution.png")

# Шаг 9: Оценка качества
print_step("ШАГ 9: ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
print("📈 Оцениваем качество кластеризации...")

# Оцениваем только не-шумовые точки
non_noise_mask = labels != -1
if sum(non_noise_mask) > 1:
    silhouette_avg = silhouette_score(embeddings_normalized[non_noise_mask], labels[non_noise_mask])
    print(f"✅ Silhouette Score: {silhouette_avg:.3f}")
else:
    print("⚠️  Недостаточно точек для оценки silhouette score")

# Выводим итоговую статистику
print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
print(f"   • Всего отзывов: {len(cleaned_reviews)}")
print(f"   • Количество кластеров: {n_clusters}")
print(f"   • Шумовые точки: {n_noise} ({n_noise / len(cleaned_reviews) * 100:.1f}%)")
print(f"   • Категорий выявлено: {len(set([info['category'] for info in cluster_info]))}")

# Выводим категории
print(f"\n🏆 КЛАСТЕРЫ:")
top_categories = sorted(cluster_info, key=lambda x: x['size'], reverse=True)[:min(10, len(cluster_info))]
for i, cat in enumerate(top_categories, 1):
    print(f"   {i}. {cat['category']}")

print_step("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
print("🎉 Все этапы выполнены. Результаты сохранены в файлы:")
print("   • banking_reviews_clustered_improved.csv - полные результаты")
print("   • clusters_info_detailed.csv - детальная информация о кластерах")
print("   • category_distribution.png - график распределения категорий")