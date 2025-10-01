# ReviewParser.py - ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ
import time
import json
import os
from selectolax.parser import HTMLParser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from DataMining_service.GEO_service.get_region import get_region_dadata
import re
from datetime import datetime
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service
import psutil


class ReviewParser:
    def __init__(self):
        self.driver = None
        self.max_retries = 3
        self.base_timeout = 20
        self.driver_service = None

    def kill_chrome_processes(self):
        """Принудительное завершение всех процессов Chrome"""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if proc.info["name"] and "chrome" in proc.info["name"].lower():
                    try:
                        if proc.info["cmdline"] and any(
                                "--headless" in arg for arg in proc.info["cmdline"]
                        ):
                            proc.terminate()
                            time.sleep(1)
                            if proc.is_running():
                                proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            print(f"Ошибка при завершении Chrome процессов: {e}")

    def setup_driver(self):
        """Создание нового WebDriver"""
        if self.driver:
            return

        self.kill_chrome_processes()
        time.sleep(2)

        try:
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-images")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-web-security")
            options.add_argument("--disable-features=VizDisplayCompositor")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            options.add_argument("--log-level=3")
            options.add_argument("--disable-logging")
            options.add_argument("--quiet")
            options.page_load_strategy = "eager"
            options.add_experimental_option(
                "prefs",
                {
                    "profile.managed_default_content_settings.images": 2,
                    "profile.default_content_setting_values.notifications": 2,
                },
            )

            self.driver_service = Service()
            self.driver = webdriver.Chrome(service=self.driver_service, options=options)
            self.driver.set_page_load_timeout(self.base_timeout)
            self.driver.implicitly_wait(5)

        except Exception as e:
            self.cleanup_driver()
            raise

    def safe_get_page(self, url, retries=3):
        """Безопасная загрузка страницы"""
        for attempt in range(retries):
            try:
                if not self.driver:
                    self.setup_driver()

                self.driver.get(url)
                return True

            except (TimeoutException, WebDriverException) as e:
                self.cleanup_driver()

                if attempt < retries - 1:
                    time.sleep(3)
                    continue
                else:
                    return False

            except Exception as e:
                self.cleanup_driver()
                if attempt == retries - 1:
                    return False
                time.sleep(2)

        return False

    def cleanup_driver(self):
        """Полная очистка WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            finally:
                self.driver = None

        if self.driver_service:
            try:
                self.driver_service.stop()
            except Exception:
                pass
            finally:
                self.driver_service = None

        self.kill_chrome_processes()

    def normalize_date_to_datetime(self, date_text):
        """Нормализация даты"""
        if not date_text:
            return datetime.now().isoformat()

        date_text = date_text.strip().lower()
        now = datetime.now()

        if any(p in date_text for p in ["только что", "сейчас", "недавно", "минуту назад"]):
            return now.isoformat()

        if re.match(r"^\d{1,2}:\d{2}$", date_text):
            time_parts = date_text.split(":")
            dt = datetime.combine(
                now.date(),
                datetime.min.time().replace(
                    hour=int(time_parts[0]), minute=int(time_parts[1])
                ),
            )
            return dt.isoformat()

        months = {
            "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
            "мая": "05", "июня": "06", "июля": "07", "августа": "08",
            "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
        }

        for m, n in months.items():
            if m in date_text:
                d = re.search(r"(\d{1,2})", date_text)
                if d:
                    day = int(d.group(1))
                    month = int(n)
                    year = now.year
                    time_match = re.search(r"(\d{1,2}):(\d{2})", date_text)
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2))
                        dt = datetime(year, month, day, hour, minute)
                    else:
                        dt = datetime(year, month, day, now.hour, now.minute)
                    return dt.isoformat()

        date_match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})", date_text)
        if date_match:
            day, month, year, hour, minute = map(int, date_match.groups())
            dt = datetime(year, month, day, hour, minute)
            return dt.isoformat()

        return now.isoformat()

    def extract_banki_review_data(self, url):
        """Извлекает текст и город с детальной страницы банки.ру"""
        try:
            if not self.safe_get_page(url, retries=self.max_retries):
                return "Ошибка загрузки страницы", None

            time.sleep(0.8)
            parser = HTMLParser(self.driver.page_source)

            text = None
            for sel in [
                '[class*="ResponseText"]',
                'div[class*="Text"] p',
                '[class*="response-text"]',
            ]:
                el = parser.css_first(sel)
                if el and len(el.text()) > 50 and "JavaScript" not in el.text():
                    text = el.text().strip()
                    break

            if not text:
                for p in parser.css("p")[:3]:
                    t = p.text().strip()
                    if len(t) > 80 and any(w in t.lower() for w in ["банк", "карт"]):
                        text = t
                        break

            if not text:
                text = "Текст не найден"

            city = None
            city_element = parser.css_first(".l3a372298")
            if city_element:
                city = city_element.text().strip()

            if not city:
                author_block = parser.css_first('[class*="lf4cbd87d"]')
                if author_block:
                    spans = author_block.css("span")
                    for span in spans:
                        span_text = span.text().strip()
                        if (
                                span_text
                                and len(span_text) < 50
                                and not span_text.startswith("user")
                                and not span_text.isdigit()
                                and re.match(r"^[А-Яа-яё\s\-]+$", span_text)
                        ):
                            city = span_text
                            break

            return text, city

        except Exception as e:
            print(f"Ошибка при извлечении данных с {url}: {repr(e)}")
            return f"Ошибка извлечения данных: {repr(e)}", None

    def get_banki_reviews(self, limit=5):
        """Получение отзывов с banki.ru"""
        try:
            if not self.safe_get_page(
                    "https://www.banki.ru/services/responses/bank/gazprombank/?type=all",
                    retries=self.max_retries,
            ):
                print("Не удалось загрузить главную страницу banki.ru")
                return []

            try:
                WebDriverWait(self.driver, 4).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, '[data-test="responses__response"]')
                    )
                )
            except TimeoutException:
                print("Элементы отзывов не найдены на banki.ru")

            parser = HTMLParser(self.driver.page_source)
            reviews = []

            for i, elem in enumerate(
                    parser.css('[data-test="responses__response"]')[:limit]
            ):
                try:
                    title = elem.css_first("h3 a")
                    if title:
                        href = title.attributes.get("href", "")
                        full_url = f"https://www.banki.ru{href}"
                        text, city = self.extract_banki_review_data(full_url)
                        region = None
                        region_code = None
                        if city:
                            region_data = get_region_dadata(city)
                            if region_data and isinstance(region_data, dict):
                                region_code = region_data.get("region_code")
                                region = region_data.get("region")

                        # Извлечение времени
                        date_match = re.search(
                            r"(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})", elem.text()
                        )
                        datetime_review = None
                        if date_match:
                            datetime_review = self.normalize_date_to_datetime(
                                date_match.group(1)
                            )

                        review = {
                            "text": text,
                            "city": city,
                            "region": region,
                            "datetime_review": datetime_review,
                            "region_code": region_code,
                            "source_id": "banki.ru",
                            "url": full_url,
                        }
                        reviews.append(review)

                except Exception as e:
                    continue

            return reviews

        except Exception as e:
            print(f"Критическая ошибка в get_banki_reviews: {e}")
            return []

    def get_sravni_reviews(self, limit=5):
        """получение отзывов с sravni.ru"""
        try:
            if not self.safe_get_page(
                    "https://www.sravni.ru/bank/gazprombank/otzyvy/?orderBy=byDate&filterBy=all",
                    retries=self.max_retries,
            ):
                print("Не удалось загрузить страницу sravni.ru")
                return []

            time.sleep(2.0)

            cards = self.driver.find_elements(
                By.CSS_SELECTOR, '[class*="review-card_wrapper"]'
            )

            reviews_data = []
            for card in cards[:limit]:
                try:
                    # Получаем URL
                    link_elem = card.find_element(
                        By.CSS_SELECTOR, 'a[href*="/otzyvy/"]'
                    )
                    review_url = link_elem.get_attribute("href")

                    city = None
                    try:
                        card_text = card.text
                        city_patterns = [
                            r'г\.?\s*([А-Яа-яё\-]+)',
                            r'([А-Яа-яё\-]{3,15})\s*\n',
                        ]

                        for pattern in city_patterns:
                            matches = re.findall(pattern, card_text)
                            if matches:
                                for match in matches:
                                    match = match.strip()
                                    if (len(match) >= 3 and
                                            not any(x in match.lower() for x in
                                                    ["банк", "отзыв", "карт", "читать", "ответ", "показать",
                                                     "газпромбанк"])):
                                        city = match
                                        break
                            if city:
                                break


                    except Exception as e:
                        print(f"Ошибка извлечения города из списка: {e}")

                    reviews_data.append({
                        "url": review_url,
                        "city": city
                    })

                except Exception as e:
                    print(f"Ошибка сбора данных из карточки: {e}")
                    continue

            reviews = []

            for idx, data in enumerate(reviews_data):
                try:
                    review_url = data["url"]
                    city = data["city"]

                    if not self.safe_get_page(review_url, retries=self.max_retries):
                        continue

                    time.sleep(1.5)

                    html_content = self.driver.page_source
                    parser = HTMLParser(html_content)
                    text = None
                    title_elem = parser.css_first('[class*="review-card_title"]')
                    if title_elem:
                        text = title_elem.text().strip()

                    full_text_elem = parser.css_first('[class*="review-card_text"] p')
                    if full_text_elem:
                        full_text = full_text_elem.text().strip()
                        if len(full_text) > len(text or ""):
                            text = full_text

                    if not text or len(text) <= 5:
                        continue

                    region = None
                    region_code = None
                    if city:
                        region_data = get_region_dadata(city)
                        if region_data and isinstance(region_data, dict):
                            region_code = region_data.get("region_code")
                            region = region_data.get("region")

                    datetime_review = None
                    time_selectors = [
                        '[class*="h-color-D30"]',
                        '[class*="date"]',
                        '[class*="time"]',
                        'time',
                    ]

                    for selector in time_selectors:
                        time_elem = parser.css_first(selector)
                        if time_elem:
                            time_text = time_elem.text().strip()
                            if time_text:
                                datetime_review = self.normalize_date_to_datetime(time_text)
                                break

                    if not datetime_review:
                        datetime_review = datetime.now().isoformat()

                    review = {
                        "text": text,
                        "city": city,
                        "region": region,
                        "datetime_review": datetime_review,
                        "region_code": region_code,
                        "source_id": "sravni.ru",
                        "url": review_url,
                    }

                    reviews.append(review)

                except Exception as e:
                    continue

            return reviews

        except Exception as e:
            return []

    def get_reviews(self, banki_limit=5, sravni_limit=5):
        """Главный метод получения отзывов"""
        try:
            banki = self.get_banki_reviews(banki_limit)
            sravni = self.get_sravni_reviews(sravni_limit)

            return {
                "banki.ru": banki,
                "sravni.ru": sravni,
                "total": len(banki) + len(sravni),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {
                "banki.ru": [],
                "sravni.ru": [],
                "total": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        finally:
            self.cleanup_driver()

    def close(self):
        """Закрытие парсера"""
        self.cleanup_driver()

    def __del__(self):
        """Деструктор"""
        self.cleanup_driver()


class ReviewMonitor:
    def __init__(self, storage_file="review_state.json"):
        self.storage_file = storage_file
        self.parser = ReviewParser()

    def load_state(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "banki_first_url": data.get("banki_first_url"),
                    "sravni_first_url": data.get("sravni_first_url"),
                }
        return {"banki_first_url": None, "sravni_first_url": None}

    def save_state(self, banki_first_url, sravni_first_url):
        data = {
            "banki_first_url": banki_first_url,
            "sravni_first_url": sravni_first_url,
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def initialize_state(self):
        """Инициализация состояния"""
        current = self.parser.get_reviews(banki_limit=10, sravni_limit=10)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"initial_reviews_{timestamp}.json"

        initial_reviews = []
        for source in ["banki.ru", "sravni.ru"]:
            for review in current[source]:
                initial_reviews.append({
                    "source_id": review["source_id"],
                    "text": review["text"],
                    "city": review["city"],
                    "region": review["region"],
                    "datetime_review": review["datetime_review"],
                    "region_code": review["region_code"],
                })

        result = {
            "reviews": initial_reviews,
            "total_initial": len(initial_reviews),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "initial_collection",
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        banki_first_url = current["banki.ru"][0]["url"] if current["banki.ru"] else None
        sravni_first_url = current["sravni.ru"][0]["url"] if current["sravni.ru"] else None
        self.save_state(banki_first_url, sravni_first_url)

        return result

    def get_new_reviews(self):
        """Проверка новых отзывов"""
        state = self.load_state()

        if state["banki_first_url"] is None and state["sravni_first_url"] is None:
            return self.initialize_state()

        current = self.parser.get_reviews()
        new_reviews = []
        has_new = False

        if current["banki.ru"]:
            current_first_banki = current["banki.ru"][0]["url"]
            if current_first_banki != state["banki_first_url"]:
                for review in current["banki.ru"]:
                    new_reviews.append({
                        "source_id": review["source_id"],
                        "text": review["text"],
                        "city": review["city"],
                        "region": review["region"],
                        "datetime_review": review["datetime_review"],
                        "region_code": review["region_code"],
                    })
                    has_new = True

        if current["sravni.ru"]:
            current_first_sravni = current["sravni.ru"][0]["url"]
            if current_first_sravni != state["sravni_first_url"]:
                for review in current["sravni.ru"]:
                    new_reviews.append({
                        "source_id": review["source_id"],
                        "text": review["text"],
                        "city": review["city"],
                        "region": review["region"],
                        "datetime_review": review["datetime_review"],
                        "region_code": review["region_code"],
                    })
                    has_new = True

        if not has_new:
            return None

        new_banki_first = (
            current["banki.ru"][0]["url"]
            if current["banki.ru"]
            else state["banki_first_url"]
        )
        new_sravni_first = (
            current["sravni.ru"][0]["url"]
            if current["sravni.ru"]
            else state["sravni_first_url"]
        )
        self.save_state(new_banki_first, new_sravni_first)

        result = {
            "reviews": new_reviews,
            "total_new": len(new_reviews),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "new_reviews",
        }
        return result

    def close(self):
        if self.parser:
            self.parser.close()


if __name__ == "__main__":
    monitor = ReviewMonitor()
    try:
        result = monitor.get_new_reviews()
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("Новых отзывов не найдено!")
    finally:
        monitor.close()
