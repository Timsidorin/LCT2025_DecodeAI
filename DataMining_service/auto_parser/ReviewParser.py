import time
import json
import os
from selectolax.parser import HTMLParser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from GEO_service.get_region import get_region_dadata
import re
from datetime import datetime
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service
import psutil
import signal


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
        """Создание нового WebDriver с улучшенной стабильностью"""
        if self.driver:
            return

        # Очищаем старые процессы
        self.kill_chrome_processes()
        time.sleep(2)

        try:
            options = Options()
            options.add_argument("--headless=new")  # Новый headless режим
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

            # Отключаем логи Chrome
            options.add_argument("--log-level=3")
            options.add_argument("--disable-logging")
            options.add_argument("--quiet")

            # Критически важные настройки для стабильности
            options.page_load_strategy = "eager"
            options.add_experimental_option(
                "prefs",
                {
                    "profile.managed_default_content_settings.images": 2,
                    "profile.default_content_setting_values.notifications": 2,
                },
            )

            # Создаем сервис с дополнительными параметрами
            self.driver_service = Service()

            self.driver = webdriver.Chrome(service=self.driver_service, options=options)
            self.driver.set_page_load_timeout(self.base_timeout)
            self.driver.implicitly_wait(5)

        except Exception as e:
            print(f"Ошибка создания WebDriver: {e}")
            self.cleanup_driver()
            raise

    def safe_get_page(self, url, retries=3):
        """Безопасная загрузка страницы с обработкой всех ошибок"""
        for attempt in range(retries):
            try:
                if not self.driver:
                    self.setup_driver()

                self.driver.get(url)
                return True

            except (TimeoutException, WebDriverException) as e:
                print(f"Ошибка соединения на попытке {attempt + 1} для {url}: {e}")

                # Пересоздаем драйвер при критических ошибках
                self.cleanup_driver()

                if attempt < retries - 1:
                    time.sleep(3)
                    continue
                else:
                    print(f"Окончательная ошибка для {url}")
                    return False

            except Exception as e:
                print(f"Неожиданная ошибка на попытке {attempt + 1}: {e}")
                self.cleanup_driver()
                if attempt == retries - 1:
                    return False
                time.sleep(3)

        return False

    def cleanup_driver(self):
        """Полная очистка WebDriver и связанных процессов"""
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

        # Принудительное завершение процессов
        self.kill_chrome_processes()

    def normalize_date_to_datetime(self, date_text):
        if not date_text:
            return None

        date_text = date_text.strip().lower()
        now = datetime.now()

        if any(
            p in date_text for p in ["только что", "сейчас", "недавно", "минуту назад"]
        ):
            return now.isoformat()

        if re.match(r"^\d{1,2}:\d{2}$", date_text):
            today = now.date()
            time_parts = date_text.split(":")
            dt = datetime.combine(
                today,
                datetime.min.time().replace(
                    hour=int(time_parts[0]), minute=int(time_parts[1])
                ),
            )
            return dt.isoformat()

        months = {
            "января": "01",
            "февраля": "02",
            "марта": "03",
            "апреля": "04",
            "мая": "05",
            "июня": "06",
            "июля": "07",
            "августа": "08",
            "сентября": "09",
            "октября": "10",
            "ноября": "11",
            "декабря": "12",
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

        date_match = re.search(
            r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})", date_text
        )
        if date_match:
            day, month, year, hour, minute = map(int, date_match.groups())
            dt = datetime(year, month, day, hour, minute)
            return dt.isoformat()

        return None

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

    def extract_sravni_city(self, card_element):
        """Извлекает город из карточки отзыва на sravni.ru"""
        try:
            card_text = card_element.text
            city_patterns = [
                r"г\.\s*([А-Яа-яё\-\s]+?)(?:\s|,|$)",
                r"город\s+([А-Яа-яё\-\s]+?)(?:\s|,|$)",
                r"\b([А-Яа-яё]{3,15})\b(?=\s*$|\s*\d|\s*[,.])",
            ]

            for pattern in city_patterns:
                matches = re.findall(pattern, card_text)
                if matches:
                    city_candidate = matches[0].strip()
                    if len(city_candidate) >= 3 and not any(
                        x in city_candidate.lower() for x in ["банк", "отзыв", "карт"]
                    ):
                        return city_candidate

            return None
        except:
            return None

    def get_banki_reviews(self, limit=5):
        """Получение отзывов с banki.ru с улучшенной обработкой ошибок"""
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
                        region_code = get_region_dadata(city) if city else None

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
                            "datetime_review": datetime_review,
                            "source_id": "banki.ru",
                            "city": city,
                            "region_code": region_code,
                            "url": full_url,
                            "position": i + 1,
                        }
                        reviews.append(review)
                except Exception as e:
                    print(f"Ошибка обработки отзыва {i}: {e}")
                    continue

            return reviews

        except Exception as e:
            print(f"Критическая ошибка в get_banki_reviews: {e}")
            return []

    def expand_text(self, card):
        try:
            links = card.find_elements(By.TAG_NAME, "a")
            for link in links:
                if link.text.strip().lower() == "читать":
                    self.driver.execute_script("arguments[0].click();", link)
                    time.sleep(0.5)
                    return True
        except:
            pass
        return False

    def get_sravni_reviews(self, limit=5):
        try:
            if not self.safe_get_page(
                "https://www.sravni.ru/bank/gazprombank/otzyvy/?orderBy=byDate&filterBy=all",
                retries=self.max_retries,
            ):
                print("Не удалось загрузить страницу sravni.ru")
                return []

            time.sleep(1.5)

            cards = self.driver.find_elements(
                By.CSS_SELECTOR, '[class*="review-card_wrapper"]'
            )
            reviews = []
            processed = 0

            for card in cards:
                if processed >= limit:
                    break

                try:
                    title_elem = card.find_element(
                        By.CSS_SELECTOR, '[class*="review-card_title"]'
                    )
                    title = title_elem.text.strip()

                    if len(title) <= 5:
                        continue

                    link_elem = card.find_element(
                        By.CSS_SELECTOR, 'a[href*="/otzyvy/"]'
                    )
                    url = link_elem.get_attribute("href")
                    city = self.extract_sravni_city(card)
                    region_code = get_region_dadata(city) if city else None

                    self.expand_text(card)

                    text = ""
                    try:
                        full_text = card.find_element(
                            By.CSS_SELECTOR,
                            '[class*="review-card_text"][class*="in-list"] span p',
                        )
                        text = full_text.text.strip()
                    except:
                        try:
                            text_elem = card.find_element(
                                By.CSS_SELECTOR, '[class*="review-card_text"] p'
                            )
                            text = text_elem.text.strip()
                        except:
                            text = title

                    datetime_review = None
                    time_elems = card.find_elements(
                        By.CSS_SELECTOR, '[class*="h-color-D30"]'
                    )
                    for time_elem in time_elems:
                        time_text = time_elem.text.strip()
                        if time_text and (
                            ":" in time_text
                            or any(
                                m in time_text.lower()
                                for m in [
                                    "января",
                                    "февраля",
                                    "марта",
                                    "апреля",
                                    "мая",
                                    "июня",
                                    "июля",
                                    "августа",
                                    "сентября",
                                    "октября",
                                    "ноября",
                                    "декабря",
                                ]
                            )
                            or any(
                                p in time_text.lower()
                                for p in ["только что", "сейчас", "недавно"]
                            )
                        ):
                            datetime_review = self.normalize_date_to_datetime(time_text)
                            break

                    if not datetime_review:
                        datetime_review = datetime.now().isoformat()

                    review = {
                        "text": text,
                        "datetime_review": datetime_review,
                        "source_id": "sravni.ru",
                        "city": city,
                        "region_code": region_code,
                        "url": url,
                        "position": processed + 1,
                    }
                    reviews.append(review)
                    processed += 1

                except Exception as e:
                    print(f"Ошибка обработки карточки на sravni.ru: {repr(e)}")
                    continue

            return reviews

        except Exception as e:
            print(f"Критическая ошибка в get_sravni_reviews: {e}")
            return []

    def get_reviews(self, banki_limit=5, sravni_limit=5):
        """Главный метод получения отзывов с полной очисткой ресурсов"""
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
            print(f"Ошибка при получении отзывов: {e}")
            return {
                "banki.ru": [],
                "sravni.ru": [],
                "total": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        finally:
            # ВСЕГДА очищаем ресурсы
            self.cleanup_driver()

    def close(self):
        """Закрытие парсера"""
        self.cleanup_driver()

    def __del__(self):
        """Деструктор для принудительной очистки"""
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
        # Собираем первые 10 отзывов с каждого сайта
        current = self.parser.get_reviews(banki_limit=10, sravni_limit=10)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"initial_reviews_{timestamp}.json"

        initial_reviews = []
        for source in ["banki.ru", "sravni.ru"]:
            for review in current[source]:
                initial_reviews.append(
                    {
                        "text": review["text"],
                        "datetime_review": review["datetime_review"],
                        "source_id": review["source_id"],
                        "city": review["city"],
                        "region_code": review["region_code"],
                    }
                )

        result = {
            "reviews": initial_reviews,
            "total_initial": len(initial_reviews),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "initial_collection",
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        banki_first_url = current["banki.ru"][0]["url"] if current["banki.ru"] else None
        sravni_first_url = (
            current["sravni.ru"][0]["url"] if current["sravni.ru"] else None
        )
        self.save_state(banki_first_url, sravni_first_url)
        return result

    def get_new_reviews(self):
        state = self.load_state()

        if state["banki_first_url"] is None and state["sravni_first_url"] is None:
            return self.initialize_state()

        current = self.parser.get_reviews()

        new_reviews = []
        has_new = False

        # Проверка banki.ru
        if current["banki.ru"]:
            current_first_banki = current["banki.ru"][0]["url"]
            if current_first_banki != state["banki_first_url"]:
                for review in current["banki.ru"]:
                    new_reviews.append(
                        {
                            "text": review["text"],
                            "datetime_review": review["datetime_review"],
                            "source_id": review["source_id"],
                            "city": review["city"],
                            "region_code": review["region_code"],
                        }
                    )
                    has_new = True

        # Проверка sravni.ru
        if current["sravni.ru"]:
            current_first_sravni = current["sravni.ru"][0]["url"]
            if current_first_sravni != state["sravni_first_url"]:
                for review in current["sravni.ru"]:
                    new_reviews.append(
                        {
                            "text": review["text"],
                            "datetime_review": review["datetime_review"],
                            "source_id": review["source_id"],
                            "city": review["city"],
                            "region_code": review["region_code"],
                        }
                    )
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
            print(result)
        else:
            print("Новых отзывов не найдено!")
    finally:
        monitor.close()
