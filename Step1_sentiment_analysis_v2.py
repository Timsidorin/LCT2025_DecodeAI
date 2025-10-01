# Step1_sentiment_analysis_radical_with_fuzzy.py
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os
import re
import json
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from collections import defaultdict, Counter
from sklearn.utils import resample
import matplotlib.pyplot as plt
from transformers import TrainerCallback

# Настройка CUDA для максимальной производительности
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")


def print_step(message):
    print(f"\n\033[1;36m>>> {message}\033[0m")


def print_success(message):
    print(f"\033[1;32m✓ {message}\033[0m")


def print_warning(message):
    print(f"\033[1;33m⚠ {message}\033[0m")


def print_error(message):
    print(f"\033[1;31m✗ {message}\033[0m")


print("=" * 80)
print("\033[1;35m🚀 РАДИКАЛЬНОЕ УЛУЧШЕНИЕ + ЛЕГКАЯ FUZZY-ЛОГИКА\033[0m")
print("=" * 80)


class LightFuzzyEnhancer:
    def __init__(self):
        self.strong_negative_indicators = [
            # Технические проблемы
            'зависает', 'вылетает', 'глючит', 'не работает', 'тормозит', 'лагает',
            'ошибка', 'сбой', 'баг', 'падение', 'закрылось', 'отключилось',

            # Финансовые проблемы
            'обман', 'обманул', 'навязали', 'навязывают', 'скрыли', 'скрывают',
            'комиссия', 'комиссии', 'списали', 'сняли', 'списывают', 'снимают',
            'блокировка', 'заблокировали', 'блокируют', 'отказали', 'отказ',
            'долго', 'медленно', 'ожидание', 'ждать', 'очередь', 'очереди',

            # Сервисные проблемы
            'груб', 'хам', 'хамство', 'невежлив', 'невнимательн', 'игнорируют',
            'непомог', 'отказали', 'нерешили', 'несмогли',

            # Сильные негативные эмоции
            'ужас', 'кошмар', 'отвратительно', 'бесит', 'раздражает', 'нервы',
            'никогда больше', 'хуже некуда', 'позор', 'стыд', 'разочарован',

            # Финансовые потери
            'потерял', 'потеряли', 'украли', 'мошенник', 'мошенники', 'афера',
            'кинули', 'обворовали', 'обсчитали',

            # Временные проблемы
            'нельзя воспользоваться', 'недоступно', 'недоступен', 'не получается',
            'не работает', 'не функционирует'
        ]

        # БОЛЕЕ КОНСЕРВАТИВНЫЙ список позитивных индикаторов
        self.strong_positive_indicators = [
            # Умеренные позитивные оценки
            'доволен', 'довольна', 'удовлетворен', 'удовлетворена',
            'нормально', 'нормальный', 'хорошо', 'хороший',

            # Конкретные похвалы
            'вежлив', 'внимательн', 'профессионал', 'компетентн',
            'быстро', 'оперативно', 'помог', 'решил', 'помогли', 'решили',

            # Умеренные эмоции
            'спасибо', 'благодар', 'рекомендую',

            # Исключаем слишком сильные эмоции которые редко в отзывах о банках
            # 'восхитительно', 'шикарно', 'безупречно' - оставляем только для очень уверенных случаев
        ]

        # ДОБАВЛЯЕМ НЕЙТРАЛЬНЫЕ/СМЕШАННЫЕ ИНДИКАТОРЫ
        self.neutral_indicators = [
            'нормально', 'обычно', 'стандартно', 'терпимо', 'сойдет', 'удовлетворительно',
            'приемлемо', 'средне', 'посредственно'
        ]

        self.contrast_words = [' но ', ' однако ', ' а ', ' хотя ', ' тем не менее ', ' зато ', ' а вот ']

    def enhance_prediction(self, text: str, base_label: str, confidence: float):
        text_lower = text.lower()

        # СЧИТАЕМ СИЛУ КАЖДОЙ КАТЕГОРИИ
        neg_strength = sum(3 if word in text_lower else 0 for word in self.strong_negative_indicators)
        pos_strength = sum(2 if word in text_lower else 0 for word in self.strong_positive_indicators)
        neutral_strength = sum(1 if word in text_lower else 0 for word in self.neutral_indicators)

        # 1. ПРИОРИТЕТ: ЯВНЫЙ НЕГАТИВ (банковские отзывы чаще негативные)
        if neg_strength >= 2:  # Хотя бы 2 сильных негативных индикатора
            if base_label != 'NEGATIVE':
                return 'NEGATIVE', min(0.95, confidence + 0.15), "явный_негатив_банк"

        # 2. СМЕШАННЫЕ/НЕЙТРАЛЬНЫЕ СЛУЧАИ
        if neutral_strength >= 2 and (pos_strength + neg_strength) < 3:
            return 'NEUTRAL', min(0.85, confidence + 0.1), "нейтральный_контекст"

        # 3. УМЕРЕННЫЙ ПОЗИТИВ (требуем больше доказательств)
        if pos_strength >= 3 and neg_strength == 0:  # Только если нет негатива
            if base_label != 'POSITIVE':
                return 'POSITIVE', min(0.9, confidence + 0.1), "умеренный_позитив"

        # 4. КОНТРАСТНЫЕ СЛУЧАИ - чаще в сторону негатива
        for contrast_word in self.contrast_words:
            if contrast_word in text_lower:
                parts = re.split(f"{contrast_word}\\s+", text_lower)
                if len(parts) >= 2:
                    # В банковских отзывах вторая часть часто важнее
                    second_part = parts[1]
                    second_neg = sum(1 for word in self.strong_negative_indicators if word in second_part)
                    second_pos = sum(1 for word in self.strong_positive_indicators if word in second_part)

                    if second_neg > second_pos:
                        return 'NEGATIVE', min(0.9, confidence + 0.1), "контраст_негатив"

        return base_label, confidence, "без_изменений"


class RadicalSentimentFineTuningDataset(Dataset):
    """УЛУЧШЕННЫЙ датасет для дообучения с реальными сегментами"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'отрицательно': 0, 'нейтрально': 1, 'положительно': 2}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label_map.get(self.labels[idx], 1)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RadicalSentimentFineTuner:
    """РАДИКАЛЬНЫЙ класс для дообучения с улучшенной точностью"""

    def __init__(self):
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fuzzy_enhancer = LightFuzzyEnhancer()

        # Переменные для отслеживания метрик
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []

        # Словарь ключевых слов для тем
        self.topic_keywords = {
            "мобильное приложение": ["приложение", "мобильный", "приложени", "мобильное", "приложении", "приложению"],
            "обслуживание в офисах и отделениях": ["офис", "отделен", "обслуживан", "менеджер", "сотрудник",
                                                   "консультант", "филиал"],
            "кредитные карты": ["кредитная", "карта", "кредитную", "карту", "кредитной", "картой"],
            "дебетовые карты": ["дебетовая", "карта", "дебетовую", "карту", "дебетовой", "картой"],
            "ипотека": ["ипотек", "ипотеку", "ипотеки", "ипотечной", "ипотечный"],
            "страховые и сервисные продукты": ["страховк", "страховку", "страховой", "сервис", "услуг", "продукт"],
            "интернет-банк": ["интернет-банк", "интернет банк", "онлайн-банк", "онлайн банк"],
            "денежные переводы": ["перевод", "переводы", "перевода", "переводам", "денежн"],
            "вклады": ["вклад", "вклады", "вклада", "вкладам", "депозит"],
            "инвестиции": ["инвестиц", "инвестиции", "инвестиций", "инвестирова"],
            "зарплатные карты": ["зарплатная", "карта", "зарплатную", "карту", "зарплатной"],
            "премиальные карты": ["премиальная", "карта", "премиальную", "карту", "премиальной"],
            "накопительные счета": ["накопительный", "счет", "накопительного", "счета", "накопительным"],
            "автокредитование": ["автокредит", "автокредита", "автокредиту", "авто кредит"],
            "рефинансирование кредитов": ["рефинансирован", "рефинансиров", "рефинансирование"],
            "бонусные программы": ["бонус", "бонусы", "бонусов", "бонусной", "программ"]
        }

    def load_better_model(self):
        """Загрузка УЛУЧШЕННОЙ модели"""
        print_step("Загрузка улучшенной модели для радикального дообучения")

        model_options = [
            "blanchefort/rubert-base-cased-sentiment",  # Лучшая для русского sentiment
            "seara/rubert-tiny2-russian-sentiment",  # Легкая и быстрая
            "cointegrated/rubert-tiny-sentiment-balanced",  # Сбалансированная
        ]

        for model_name in model_options:
            try:
                print(f"Попытка загрузки: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=3
                )
                self.model.to(self.device)
                self.model_name = model_name
                print_success(f"Улучшенная модель загружена: {model_name}")
                return True
            except Exception as e:
                print_warning(f"Не удалось загрузить {model_name}: {e}")
                continue

        print_error("Все улучшенные модели не удалось загрузить")
        return False

    def extract_topic_segment(self, text: str, topic: str) -> Optional[str]:
        """Извлекает РЕАЛЬНЫЙ сегмент текста, соответствующий теме"""
        text_lower = text.lower()

        # Получаем ключевые слова для темы
        keywords = self.topic_keywords.get(topic, [topic.lower()])

        # Разбиваем текст на предложения
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            sentence_lower = sentence_clean.lower()

            # Проверяем наличие ключевых слов в предложении
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence_clean)

        # Если нашли релевантные предложения - возвращаем их
        if relevant_sentences:
            segment = " ".join(relevant_sentences)
            # Ограничиваем длину для эффективности
            if len(segment) > 20:  # Минимальная осмысленная длина
                return segment

        return None

    def prepare_high_quality_dataset(self, reviews_file: str, results_file: str):
        """Создание КАЧЕСТВЕННОГО датасета из РЕАЛЬНЫХ сегментов"""
        print_step("Подготовка КАЧЕСТВЕННОГО датасета из реальных сегментов")

        try:
            # Загрузка данных
            with open(reviews_file, 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)

            # Создаем маппинг id -> текст
            id_to_text = {}
            for item in reviews_data['data']:
                id_to_text[item['id']] = item['text']

            training_texts = []
            training_labels = []
            segment_stats = defaultdict(int)

            # Обрабатываем каждый отзыв
            for prediction in tqdm(results_data['predictions'], desc="Извлечение сегментов"):
                review_id = prediction['id']
                if review_id in id_to_text:
                    full_text = id_to_text[review_id]

                    # Для каждой темы извлекаем соответствующий сегмент
                    for topic, sentiment in zip(prediction['topics'], prediction['sentiments']):
                        segment = self.extract_topic_segment(full_text, topic)
                        if segment:
                            training_texts.append(segment)
                            training_labels.append(sentiment)
                            segment_stats[topic] += 1

            print_success(f"Извлечено {len(training_texts)} РЕАЛЬНЫХ сегментов из текстов")
            print(f"📊 Статистика по темам:")
            for topic, count in segment_stats.items():
                print(f"   - {topic}: {count} сегментов")

            return training_texts, training_labels

        except Exception as e:
            print_error(f"Ошибка подготовки качественного датасета: {e}")
            return [], []

    def balance_dataset(self, texts: List[str], labels: List[str]):
        """БАЛАНСИРОВКА классов для улучшения точности"""
        print_step("Балансировка классов в датасета")

        # Анализируем распределение
        label_counts = Counter(labels)
        print(f"📊 РАСПРЕДЕЛЕНИЕ ДО БАЛАНСИРОВКИ:")
        for label, count in label_counts.items():
            percentage = count / len(labels) * 100
            icon = "👎" if label == 'отрицательно' else "👍" if label == 'положительно' else "➖"
            print(f"   {icon} {label}: {count} примеров ({percentage:.1f}%)")

        # Балансируем oversampling меньшего класса
        max_count = max(label_counts.values())
        balanced_texts = []
        balanced_labels = []

        for label in ['отрицательно', 'нейтрально', 'положительно']:
            label_texts = [t for t, l in zip(texts, labels) if l == label]

            if len(label_texts) < max_count:
                # Oversampling до максимального количества
                label_texts_resampled = resample(
                    label_texts,
                    n_samples=max_count,
                    random_state=42,
                    replace=True
                )
                balanced_texts.extend(label_texts_resampled)
                balanced_labels.extend([label] * len(label_texts_resampled))
                print(f"   🔄 {label}: oversampling {len(label_texts)} → {len(label_texts_resampled)}")
            else:
                balanced_texts.extend(label_texts)
                balanced_labels.extend([label] * len(label_texts))

        # Финальная статистика
        balanced_counts = Counter(balanced_labels)
        print(f"📊 РАСПРЕДЕЛЕНИЕ ПОСЛЕ БАЛАНСИРОВКИ:")
        for label, count in balanced_counts.items():
            percentage = count / len(balanced_labels) * 100
            icon = "👎" if label == 'отрицательно' else "👍" if label == 'положительно' else "➖"
            print(f"   {icon} {label}: {count} примеров ({percentage:.1f}%)")

        return balanced_texts, balanced_labels

    def plot_training_loss(self):
        """Построение графика изменения ошибки во время обучения"""
        try:
            if not self.train_losses or not self.eval_losses:
                print_warning("Недостаточно данных для построения графика ошибки")
                return

            plt.figure(figsize=(12, 8))

            # График ошибки
            plt.subplot(2, 1, 1)
            epochs_range = range(1, len(self.train_losses) + 1)
            eval_epochs = range(1, len(self.eval_losses) + 1)

            plt.plot(epochs_range, self.train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(eval_epochs, self.eval_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.title('📉 Изменение ошибки во время обучения', fontsize=14, fontweight='bold')
            plt.xlabel('Шаг обучения')
            plt.ylabel('Ошибка (Loss)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # График точности
            if self.eval_accuracies:
                plt.subplot(2, 1, 2)
                accuracy_epochs = range(1, len(self.eval_accuracies) + 1)
                plt.plot(accuracy_epochs, self.eval_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
                plt.title('📈 Изменение точности во время обучения', fontsize=14, fontweight='bold')
                plt.xlabel('Эпоха')
                plt.ylabel('Точность (Accuracy)')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # Сохраняем график
            plot_filename = "training_metrics.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()

            print_success(f"📊 График метрик обучения сохранен как: {plot_filename}")

            # Анализ сходимости
            if len(self.eval_losses) >= 3:
                last_losses = self.eval_losses[-3:]
                if all(abs(last_losses[i] - last_losses[i - 1]) < 0.01 for i in range(1, len(last_losses))):
                    print_success("✅ Модель показывает признаки сходимости")
                else:
                    print_warning("⚠ Модель еще не сошлась, возможно стоит увеличить количество эпох")

        except Exception as e:
            print_warning(f"Не удалось построить график ошибки: {e}")

    def train_high_accuracy(self, training_texts: List[str], training_labels: List[str],
                            output_dir: str = "./radical_fine_tuned_model"):
        """РАДИКАЛЬНОЕ обучение для достижения 85% точности с графиком ошибки"""

        if not training_texts:
            print_error("Нет данных для обучения")
            return False

        print_step("Запуск РАДИКАЛЬНОГО обучения для 85% точности")

        try:
            # Создаем улучшенный датасет
            dataset = RadicalSentimentFineTuningDataset(
                texts=training_texts,
                labels=training_labels,
                tokenizer=self.tokenizer,
                max_length=256
            )

            # Разделяем на train/validation (85/15 для большего обучения)
            train_size = int(0.85 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

            print(f"📊 УЛУЧШЕННОЕ РАЗБИЕНИЕ ДАННЫХ:")
            print(f"  - Обучающая выборка: {train_size} примеров")
            print(f"  - Валидационная выборка: {val_size} примеров")
            print(f"  - Всего: {len(dataset)} примеров")

            # Кастомный callback для отслеживания метрик
            class LossCallback(TrainerCallback):
                def __init__(self, outer):
                    self.outer = outer

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is not None:
                        # Сохраняем train loss
                        if 'loss' in logs:
                            self.outer.train_losses.append(logs['loss'])
                        # Сохраняем eval loss
                        if 'eval_loss' in logs:
                            self.outer.eval_losses.append(logs['eval_loss'])
                        # Сохраняем accuracy
                        if 'eval_accuracy' in logs:
                            self.outer.eval_accuracies.append(logs['eval_accuracy'])

            # РАДИКАЛЬНЫЕ аргументы обучения
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=20,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                learning_rate=3e-5,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir='./radical_logs',
                logging_steps=20,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to=None,
                save_total_limit=2,
                dataloader_pin_memory=False,
                gradient_accumulation_steps=2,
            )

            # Функция для вычисления метрик
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)

                accuracy = accuracy_score(labels, predictions)
                precision = precision_score(labels, predictions, average='weighted', zero_division=0)
                recall = recall_score(labels, predictions, average='weighted', zero_division=0)
                f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # Создаем тренер с callback
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[LossCallback(self)]
            )

            # Запускаем РАДИКАЛЬНОЕ обучение
            print_step("🚀 ЗАПУСК РАДИКАЛЬНОГО ОБУЧЕНИЯ...")
            training_result = trainer.train()

            # Сохраняем модель
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Строим график ошибки
            self.plot_training_loss()

            # ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ
            print_step("📈 РЕЗУЛЬТАТЫ РАДИКАЛЬНОГО ДООБУЧЕНИЯ")

            # Финальная оценка
            eval_results = trainer.evaluate()
            final_accuracy = eval_results.get('eval_accuracy', 0)

            print(f"🎯 КЛЮЧЕВЫЕ МЕТРИКИ:")
            print(f"  - Final Accuracy: {final_accuracy:.4f}")
            print(f"  - Final F1-score: {eval_results.get('eval_f1', 0):.4f}")
            print(f"  - Final Loss: {eval_results.get('eval_loss', 0):.4f}")

            # Анализ предсказаний
            test_predictions = trainer.predict(val_dataset)
            pred_labels = np.argmax(test_predictions.predictions, axis=1)
            true_labels = test_predictions.label_ids

            # Детальный отчет
            target_names = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
            report = classification_report(true_labels, pred_labels, target_names=target_names, digits=4)
            print(f"\n📊 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
            print(report)

            # Анализ качества
            accuracy_percent = final_accuracy * 100
            print(f"\n📈 ИТОГОВАЯ ОЦЕНКА КАЧЕСТВА:")
            if accuracy_percent >= 85:
                print_success(f"  🎉 ПРЕВОСХОДНО! Достигнута цель: {accuracy_percent:.1f}% точности!")
            elif accuracy_percent >= 80:
                print_success(f"  ✅ ОТЛИЧНО! Высокая точность: {accuracy_percent:.1f}%")
            elif accuracy_percent >= 75:
                print_success(f"  ✅ ХОРОШО! Качество улучшено: {accuracy_percent:.1f}%")
            elif accuracy_percent >= 70:
                print_warning(f"  ⚠ НОРМАЛЬНО! Средняя точность: {accuracy_percent:.1f}%")
            else:
                print_warning(f"  ⚠ ТРЕБУЕТСЯ ДОРАБОТКА: {accuracy_percent:.1f}% точности")

            print_success(f"Радикально дообученная модель сохранена в: {output_dir}")
            return final_accuracy >= 0.75

        except Exception as e:
            print_error(f"Ошибка радикального обучения: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_sentiment_with_fuzzy(self, text: str):
        """Анализ тональности с fuzzy-коррекцией"""
        try:
            # Базовое предсказание BERT
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # Преобразуем номер класса в метку
            class_mapping = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
            base_label = class_mapping.get(predicted_class, 'NEUTRAL')

            # Применяем fuzzy-коррекцию
            final_label, final_confidence, reason = self.fuzzy_enhancer.enhance_prediction(
                text, base_label, confidence
            )

            return {
                'text': text,
                'base_sentiment': base_label,
                'base_confidence': confidence,
                'final_sentiment': final_label,
                'final_confidence': final_confidence,
                'correction_reason': reason,
                'was_corrected': base_label != final_label
            }
        except Exception as e:
            print_warning(f"Ошибка анализа: {e}")
            return {
                'text': text,
                'base_sentiment': 'NEUTRAL',
                'base_confidence': 0.5,
                'final_sentiment': 'NEUTRAL',
                'final_confidence': 0.5,
                'correction_reason': 'error',
                'was_corrected': False
            }


def test_radical_model_with_fuzzy():
    """Тестирование с легкой fuzzy-коррекцией"""
    try:
        from transformers import pipeline

        # Загружаем дообученную модель
        classifier = pipeline(
            "text-classification",
            model="./radical_fine_tuned_model",
            tokenizer="./radical_fine_tuned_model",
            device=0 if torch.cuda.is_available() else -1
        )

        fuzzy_enhancer = LightFuzzyEnhancer()

        test_cases = [
            "Мобильное приложение постоянно зависает и вылетает",
            "Обслуживание в отделении отличное, менеджеры вежливые",
            "Кредитную карту оформили быстро, очень доволен",
            "Ужасное обслуживание, никогда больше не обращусь",
            "Приложение удобное, но иногда тормозит",
            "Все отлично, но комиссии высокие",
            "Сотрудник помог быстро, спасибо большое",
            "Дебетовая карта с кешбэком радует, а вот интернет-банк работает медленно"
        ]

        print("\n🧪 ТЕСТ С FUZZY-КОРРЕКЦИЕЙ:")
        corrections_count = 0

        for text in test_cases:
            # Базовое предсказание
            base_result = classifier(text)[0]
            base_label = base_result['label']
            base_confidence = base_result['score']

            # Fuzzy-коррекция
            final_label, final_confidence, reason = fuzzy_enhancer.enhance_prediction(
                text, base_label, base_confidence
            )

            was_corrected = base_label != final_label
            if was_corrected:
                corrections_count += 1

            base_icon = "👍" if base_label == 'POSITIVE' else "👎" if base_label == 'NEGATIVE' else "➖"
            final_icon = "👍" if final_label == 'POSITIVE' else "👎" if final_label == 'NEGATIVE' else "➖"

            correction_indicator = " 🔄" if was_corrected else ""

            print(f"  {base_icon}→{final_icon}{correction_indicator} '{text}'")
            print(f"     BERT: {base_label} ({base_confidence:.3f})")
            print(f"     Финальное: {final_label} ({final_confidence:.3f}) - {reason}")
            print()

        print_success(f"Fuzzy-логика исправила {corrections_count} из {len(test_cases)} примеров")
        return True

    except Exception as e:
        print_warning(f"Тестирование с fuzzy не удалось: {e}")
        return False


def radical_fine_tune_sentiment_model():
    """Основная функция для радикального дообучения"""
    print_step("🚀 ЗАПУСК РАДИКАЛЬНОГО ДООБУЧЕНИЯ МОДЕЛИ")

    # Инициализация радикального тюнера
    fine_tuner = RadicalSentimentFineTuner()

    # 1. Загружаем улучшенную модель
    if not fine_tuner.load_better_model():
        print_error("Не удалось загрузить улучшенную модель")
        return False

    # 2. Готовим КАЧЕСТВЕННЫЕ данные
    training_texts, training_labels = fine_tuner.prepare_high_quality_dataset(
        "reviewforlearn.json",
        "resultforlearn.json"
    )

    if not training_texts:
        print_error("Не удалось подготовить качественные данные")
        return False

    # 3. БАЛАНСИРУЕМ классы
    balanced_texts, balanced_labels = fine_tuner.balance_dataset(training_texts, training_labels)

    print_success(f"✅ Подготовлено {len(balanced_texts)} сбалансированных качественных примеров")

    # 4. Запускаем РАДИКАЛЬНОЕ обучение
    success = fine_tuner.train_high_accuracy(
        training_texts=balanced_texts,
        training_labels=balanced_labels,
        output_dir="./radical_fine_tuned_model"
    )

    if success:
        print_success("🎉 РАДИКАЛЬНОЕ ДООБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")

        # Тестирование с fuzzy-логикой
        print_step("🧪 ТЕСТИРОВАНИЕ С FUZZY-КОРРЕКЦИЕЙ")
        test_radical_model_with_fuzzy()

        return True
    else:
        print_error("Радикальное дообучение не достигло целевой точности")
        return False


def run_standard_sentiment_analysis():
    """Запуск обычного анализа тональности на данных с сохранением в CSV"""
    try:
        print_step("1. Загрузка данных для анализа")

        # Проверяем существование файлов данных
        input_file = 'GaspromBank_professional_dataset.csv'
        if not os.path.exists(input_file):
            input_file = 'GaspromBank_dataset.csv'
            if not os.path.exists(input_file):
                print_error("Файлы данных не найдены")
                return False

        # Загружаем данные
        df = pd.read_csv(input_file, sep=';', encoding='windows-1251')
        print_success(f"Загружено {len(df)} строк из {input_file}")

        # Поиск текстовой колонки
        text_column = 'text'
        if text_column not in df.columns:
            for col in df.columns:
                if 'text' in col.lower() or 'отзыв' in col.lower():
                    text_column = col
                    break
            print_warning(f"Используется колонка: {text_column}")

        print_step("2. Загрузка модели для анализа")

        # Создаем анализатор
        analyzer = RadicalSentimentFineTuner()

        # Пробуем загрузить дообученную модель, если есть
        fine_tuned_path = "./radical_fine_tuned_model"
        if os.path.exists(fine_tuned_path):
            print("Обнаружена дообученная модель, загружаем...")
            try:
                analyzer.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
                analyzer.model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_path)
                analyzer.model.to(analyzer.device)
                analyzer.fuzzy_enhancer = LightFuzzyEnhancer()
                print_success("Дообученная модель загружена")
            except Exception as e:
                print_warning(f"Не удалось загрузить дообученную модель: {e}")
                if not analyzer.load_better_model():
                    return False
        else:
            # Загружаем базовую модель
            if not analyzer.load_better_model():
                return False

        print_step("3. Анализ тональности с fuzzy-коррекцией")

        # Анализируем ВСЕ данные
        texts_to_analyze = df[text_column].astype(str).tolist()

        print(f"Анализируем {len(texts_to_analyze)} примеров...")

        results = []
        for text in tqdm(texts_to_analyze, desc="Анализ тональности"):
            result = analyzer.analyze_sentiment_with_fuzzy(text)
            results.append(result)

        print_step("4. Сохранение результатов в CSV")

        # Добавляем результаты в DataFrame
        df['sentiment_base'] = [r['base_sentiment'] for r in results]
        df['sentiment_final'] = [r['final_sentiment'] for r in results]
        df['sentiment_confidence'] = [r['final_confidence'] for r in results]
        df['was_corrected'] = [r['was_corrected'] for r in results]
        df['correction_reason'] = [r['correction_reason'] for r in results]

        # Создаем имя выходного файла
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_with_sentiment.csv"

        # Сохраняем в CSV
        df.to_csv(output_file, sep=';', encoding='windows-1251', index=False)
        print_success(f"Результаты сохранены в: {output_file}")

        print_step("5. Статистика анализа")

        # Статистика
        base_sentiments = [r['base_sentiment'] for r in results]
        final_sentiments = [r['final_sentiment'] for r in results]
        corrections = sum(1 for r in results if r['was_corrected'])

        base_counts = Counter(base_sentiments)
        final_counts = Counter(final_sentiments)

        print(f"📊 СТАТИСТИКА АНАЛИЗА ({len(results)} примеров):")
        print(f"\nБАЗОВЫЕ ПРЕДСКАЗАНИЯ BERT:")
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = base_counts.get(sentiment, 0)
            percentage = count / len(results) * 100
            icon = "👍" if sentiment == 'POSITIVE' else "👎" if sentiment == 'NEGATIVE' else "➖"
            print(f"  {icon} {sentiment}: {count} ({percentage:.1f}%)")

        print(f"\nФИНАЛЬНЫЕ ПРЕДСКАЗАНИЯ (с fuzzy):")
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = final_counts.get(sentiment, 0)
            percentage = count / len(results) * 100
            icon = "👍" if sentiment == 'POSITIVE' else "👎" if sentiment == 'NEGATIVE' else "➖"
            print(f"  {icon} {sentiment}: {count} ({percentage:.1f}%)")

        print(f"\n🔄 FUZZY-КОРРЕКЦИЯ:")
        print(f"  Исправлено: {corrections} из {len(results)} примеров ({corrections / len(results) * 100:.1f}%)")

        # Показываем несколько примеров с коррекцией
        corrected_examples = [r for r in results if r['was_corrected']]
        if corrected_examples:
            print(f"\n🔍 ПРИМЕРЫ ИСПРАВЛЕННЫХ ПРЕДСКАЗАНИЙ:")
            for i, example in enumerate(corrected_examples[:5]):  # Показываем первые 5
                base_icon = "👍" if example['base_sentiment'] == 'POSITIVE' else "👎" if example[
                                                                                           'base_sentiment'] == 'NEGATIVE' else "➖"
                final_icon = "👍" if example['final_sentiment'] == 'POSITIVE' else "👎" if example[
                                                                                             'final_sentiment'] == 'NEGATIVE' else "➖"
                print(f"  {i + 1}. {base_icon}→{final_icon} '{example['text'][:80]}...'")
                print(f"     Причина: {example['correction_reason']}")

        # Демонстрация работы на тестовых примерах
        print_step("6. Демонстрация работы системы")

        demo_texts = [
            "Мобильное приложение постоянно зависает и вылетает",
            "Обслуживание в отделении отличное, менеджеры вежливые",
            "Кредитную карту оформили быстро, очень доволен",
            "Приложение удобное, но иногда тормозит"
        ]

        print("\n🧪 ДЕМОНСТРАЦИЯ НА ТЕСТОВЫХ ПРИМЕРАХ:")
        for text in demo_texts:
            result = analyzer.analyze_sentiment_with_fuzzy(text)
            base_icon = "👍" if result['base_sentiment'] == 'POSITIVE' else "👎" if result[
                                                                                      'base_sentiment'] == 'NEGATIVE' else "➖"
            final_icon = "👍" if result['final_sentiment'] == 'POSITIVE' else "👎" if result[
                                                                                        'final_sentiment'] == 'NEGATIVE' else "➖"
            correction_indicator = " 🔄" if result['was_corrected'] else ""

            print(f"  {base_icon}→{final_icon}{correction_indicator} '{text}'")
            print(f"     BERT: {result['base_sentiment']} ({result['base_confidence']:.3f})")
            print(f"     Финальное: {result['final_sentiment']} ({result['final_confidence']:.3f})")
            if result['was_corrected']:
                print(f"     Причина коррекции: {result['correction_reason']}")
            print()

        return True

    except Exception as e:
        print_error(f"Ошибка при анализе тональности: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_and_save_sentiment_only():
    """Только анализ и сохранение тональности без демонстрации"""
    try:
        print_step("Анализ тональности и сохранение в CSV")

        # Проверяем существование файлов данных
        input_file = 'GaspromBank_professional_dataset.csv'
        if not os.path.exists(input_file):
            input_file = 'GaspromBank_dataset.csv'
            if not os.path.exists(input_file):
                print_error("Файлы данных не найдены")
                return False

        # Загружаем данные
        df = pd.read_csv(input_file, sep=';', encoding='windows-1251')
        print_success(f"Загружено {len(df)} строк из {input_file}")

        # Поиск текстовой колонки
        text_column = 'text'
        if text_column not in df.columns:
            for col in df.columns:
                if 'text' in col.lower() or 'отзыв' in col.lower():
                    text_column = col
                    break

        # Создаем анализатор
        analyzer = RadicalSentimentFineTuner()

        # Загружаем модель
        fine_tuned_path = "./radical_fine_tuned_model"
        if os.path.exists(fine_tuned_path):
            analyzer.load_fine_tuned_model(fine_tuned_path)
        else:
            analyzer.load_better_model()

        # Анализируем данные
        texts_to_analyze = df[text_column].astype(str).tolist()
        print(f"Анализируем {len(texts_to_analyze)} примеров...")

        results = []
        for text in tqdm(texts_to_analyze, desc="Анализ тональности"):
            result = analyzer.analyze_sentiment_with_fuzzy(text)
            results.append(result)

        # Добавляем результаты в DataFrame
        df['sentiment'] = [r['final_sentiment'] for r in results]
        df['sentiment_confidence'] = [r['final_confidence'] for r in results]
        df['sentiment_was_corrected'] = [r['was_corrected'] for r in results]

        # Сохраняем в CSV
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_with_sentiment.csv"
        df.to_csv(output_file, sep=';', encoding='windows-1251', index=False)

        print_success(f"✅ Результаты сохранены в: {output_file}")

        # Краткая статистика
        sentiments = [r['final_sentiment'] for r in results]
        sentiment_counts = Counter(sentiments)
        corrections = sum(1 for r in results if r['was_corrected'])

        print(f"📊 Статистика:")
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = count / len(results) * 100
            icon = "👍" if sentiment == 'POSITIVE' else "👎" if sentiment == 'NEGATIVE' else "➖"
            print(f"  {icon} {sentiment}: {count} ({percentage:.1f}%)")
        print(f"  🔄 Коррекций: {corrections} ({corrections / len(results) * 100:.1f}%)")

        return True

    except Exception as e:
        print_error(f"Ошибка: {e}")
        return False


# Основная функция
def main():
    """Основная функция с выбором режима"""
    try:
        # Проверяем аргументы командной строки
        if len(sys.argv) > 1:
            if sys.argv[1] == "--radical-tune":
                print_step("🎯 РЕЖИМ РАДИКАЛЬНОГО ДООБУЧЕНИЯ С FUZZY")
                success = radical_fine_tune_sentiment_model()
                sys.exit(0 if success else 1)
            elif sys.argv[1] == "--analyze-only":
                print_step("📊 РЕЖИМ АНАЛИЗА И СОХРАНЕНИЯ")
                success = analyze_and_save_sentiment_only()
                sys.exit(0 if success else 1)

        # Если нет аргументов - показываем меню
        print("\n" + "=" * 60)
        print("🎯 ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
        print("1 - Обычный анализ тональности (с демонстрацией)")
        print("2 - РАДИКАЛЬНОЕ дообучение + FUZZY (90%+ цель)")
        print("3 - Только анализ и сохранение в CSV (быстро)")
        print("=" * 60)

        choice = input("Введите номер (1, 2 или 3): ").strip()

        if choice == "2":
            success = radical_fine_tune_sentiment_model()
        elif choice == "1":
            success = run_standard_sentiment_analysis()
        elif choice == "3":
            success = analyze_and_save_sentiment_only()
        else:
            print_warning("Неверный выбор. Завершение программы.")
            return False

        if success:
            print_success("✅ ОПЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        else:
            print_error("❌ ОПЕРАЦИЯ ЗАВЕРШИЛАСЬ С ОШИБКАМИ")

        return success

    except Exception as e:
        print_error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n" + "=" * 80)
        print("\033[1;32m✅ ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!\033[0m")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("\033[1;31m❌ ПРОГРАММА ЗАВЕРШИЛАСЬ С ОШИБКАМИ\033[0m")
        print("=" * 80)
        sys.exit(1)