# services/GenderDetector.py
import re
from typing import Optional


class GenderDetector:
    """
    Сервис для определения пола автора по тексту отзыва
    на основе окончаний русских глаголов в прошедшем времени
    """

    def __init__(self):
        self.female_markers = [
            r'\b\w+ла\b',  # любые глаголы с -ла (была, пришла, сказала)
            r'\b(была|пришла|сказала|написала|позвонила|обратилась|открыла|закрыла)\b',
            r'\b(получила|отправила|заказала|купила|взяла|попросила|хотела|могла)\b',
            r'\b(решила|подумала|увидела|услышала|узнала|поняла|осталась)\b',
        ]

        self.male_markers = [
            r'\b\w+ил\b',
            r'\b(был|пришёл|пришел|сказал|написал|позвонил|обратился|открыл|закрыл)\b',
            r'\b(получил|отправил|заказал|купил|взял|попросил|хотел|мог)\b',
            r'\b(решил|подумал|увидел|услышал|узнал|понял|остался)\b',
        ]

        self.exclude_words = [
            'была', 'были', 'будет', 'будут', 'может', 'могут',
            'стала', 'стали', 'стал', 'начала', 'начали', 'начал'
        ]

    def detect_gender(self, text: str) -> Optional[str]:
        """
        Определяет пол автора по тексту
        """
        if not text:
            return None

        text_lower = text.lower()

        female_count = 0
        for pattern in self.female_markers:
            matches = re.findall(pattern, text_lower)
            female_count += len(matches)

        male_count = 0
        for pattern in self.male_markers:
            matches = re.findall(pattern, text_lower)
            male_count += len(matches)

        if female_count > male_count and female_count >= 2:
            return "Ж"
        elif male_count > female_count and male_count >= 2:
            return "М"
        else:
            return None

    def detect_gender_simple(self, text: str) -> Optional[str]:
        """
        Ищет только явные маркеры в начале предложений
        """
        if not text:
            return None

        text_lower = text.lower()

        female_phrases = ['я была', 'я пришла', 'я обратилась', 'я получила', 'я взяла']
        male_phrases = ['я был', 'я пришёл', 'я пришел', 'я обратился', 'я получил', 'я взял']

        has_female = any(phrase in text_lower for phrase in female_phrases)
        has_male = any(phrase in text_lower for phrase in male_phrases)

        if has_female and not has_male:
            return "Ж"
        elif has_male and not has_female:
            return "М"
        else:
            return self.detect_gender(text)
