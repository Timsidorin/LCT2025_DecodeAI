import time
import json
import os
from selectolax.parser import HTMLParser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from datetime import datetime


class ReviewParser:
    def __init__(self):
        self.driver = None

    def setup_driver(self):
        if self.driver:
            return
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-images")
        options.add_argument("--disable-extensions")
        options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(6)

    def normalize_date_to_datetime(self, date_text):
        if not date_text:
            return None

        date_text = date_text.strip().lower()
        now = datetime.now()

        if any(p in date_text for p in ["только что", "сейчас", "недавно", "минуту назад"]):
            return now.isoformat()

        if re.match(r'^\d{1,2}:\d{2}$', date_text):
            today = now.date()
            time_parts = date_text.split(':')
            dt = datetime.combine(today,
                                  datetime.min.time().replace(hour=int(time_parts[0]), minute=int(time_parts[1])))
            return dt.isoformat()

        months = {'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04', 'мая': '05', 'июня': '06',
                  'июля': '07', 'августа': '08', 'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'}

        for m, n in months.items():
            if m in date_text:
                d = re.search(r'(\d{1,2})', date_text)
                if d:
                    day = int(d.group(1))
                    month = int(n)
                    year = now.year
                    time_match = re.search(r'(\d{1,2}):(\d{2})', date_text)
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2))
                        dt = datetime(year, month, day, hour, minute)
                    else:
                        dt = datetime(year, month, day, now.hour, now.minute)
                    return dt.isoformat()

        date_match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})', date_text)
        if date_match:
            day, month, year, hour, minute = map(int, date_match.groups())
            dt = datetime(year, month, day, hour, minute)
            return dt.isoformat()

        return None

    def extract_text(self, url):
        try:
            self.driver.get(url)
            time.sleep(0.8)
            parser = HTMLParser(self.driver.page_source)

            for sel in ['[class*="ResponseText"]', 'div[class*="Text"] p']:
                el = parser.css_first(sel)
                if el and len(el.text()) > 50 and 'JavaScript' not in el.text():
                    return el.text().strip()

            for p in parser.css('p')[:3]:
                t = p.text().strip()
                if len(t) > 80 and any(w in t.lower() for w in ['банк', 'карт']):
                    return t
            return "Текст не найден"
        except:
            return "Ошибка извлечения текста"

    def get_banki_reviews(self, limit=5):
        self.setup_driver()
        self.driver.get("https://www.banki.ru/services/responses/bank/gazprombank/?type=all")

        try:
            WebDriverWait(self.driver, 4).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-test="responses__response"]')))
        except:
            pass

        parser = HTMLParser(self.driver.page_source)
        reviews = []

        for i, elem in enumerate(parser.css('[data-test="responses__response"]')[:limit]):
            title = elem.css_first('h3 a')
            if title:
                href = title.attributes.get('href', '')
                full_url = f"https://www.banki.ru{href}"
                text = self.extract_text(full_url)

                date_match = re.search(r'(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})', elem.text())
                datetime_review = None
                if date_match:
                    datetime_review = self.normalize_date_to_datetime(date_match.group(1))

                review = {
                    'text': text,
                    'datetime_review': datetime_review,
                    'source_id': 'banki.ru',
                    'url': full_url,
                    'position': i + 1
                }
                reviews.append(review)
        return reviews

    def expand_text(self, card):
        try:
            links = card.find_elements(By.TAG_NAME, 'a')
            for link in links:
                if link.text.strip().lower() == 'читать':
                    self.driver.execute_script("arguments[0].click();", link)
                    time.sleep(0.5)
                    return True
        except:
            pass
        return False

    def get_sravni_reviews(self, limit=5):
        self.setup_driver()
        self.driver.get("https://www.sravni.ru/bank/gazprombank/otzyvy/?orderBy=byDate&filterBy=all")
        time.sleep(1.5)

        cards = self.driver.find_elements(By.CSS_SELECTOR, '[class*="review-card_wrapper"]')
        reviews = []
        processed = 0

        for card in cards:
            if processed >= limit:
                break

            try:
                title_elem = card.find_element(By.CSS_SELECTOR, '[class*="review-card_title"]')
                title = title_elem.text.strip()

                if len(title) <= 5:
                    continue

                link_elem = card.find_element(By.CSS_SELECTOR, 'a[href*="/otzyvy/"]')
                url = link_elem.get_attribute('href')

                self.expand_text(card)

                text = ""
                try:
                    full_text = card.find_element(By.CSS_SELECTOR,
                                                  '[class*="review-card_text"][class*="in-list"] span p')
                    text = full_text.text.strip()
                except:
                    try:
                        text_elem = card.find_element(By.CSS_SELECTOR, '[class*="review-card_text"] p')
                        text = text_elem.text.strip()
                    except:
                        text = title

                datetime_review = None
                time_elems = card.find_elements(By.CSS_SELECTOR, '[class*="h-color-D30"]')
                for time_elem in time_elems:
                    time_text = time_elem.text.strip()
                    if time_text and (':' in time_text or any(m in time_text.lower() for m in
                                                              ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                                                               'июля', 'августа', 'сентября', 'октября', 'ноября',
                                                               'декабря']) or any(
                        p in time_text.lower() for p in ['только что', 'сейчас', 'недавно'])):
                        datetime_review = self.normalize_date_to_datetime(time_text)
                        break

                if not datetime_review:
                    datetime_review = datetime.now().isoformat()

                review = {
                    'text': text,
                    'datetime_review': datetime_review,
                    'source_id': 'sravni.ru',
                    'url': url,
                    'position': processed + 1
                }
                reviews.append(review)
                processed += 1

            except:
                continue

        return reviews

    def get_reviews(self, banki_limit=5, sravni_limit=5):
        banki = self.get_banki_reviews(banki_limit)
        sravni = self.get_sravni_reviews(sravni_limit)

        if self.driver:
            self.driver.quit()

        return {
            'banki.ru': banki,
            'sravni.ru': sravni,
            'total': len(banki) + len(sravni),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def close(self):
        if self.driver:
            self.driver.quit()


class ReviewMonitor:
    def __init__(self, storage_file="review_state.json"):
        self.storage_file = storage_file
        self.parser = ReviewParser()

    def load_state(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'banki_first_url': data.get('banki_first_url'),
                    'sravni_first_url': data.get('sravni_first_url')
                }
        return {
            'banki_first_url': None,
            'sravni_first_url': None
        }

    def save_state(self, banki_first_url, sravni_first_url):
        data = {
            'banki_first_url': banki_first_url,
            'sravni_first_url': sravni_first_url,
            'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_new_reviews(self):
        state = self.load_state()
        current = self.parser.get_reviews()

        new_reviews = []
        has_new = False

        # Проверка banki.ru - если первый URL изменился, берем все текущие отзывы как новые
        if current['banki.ru']:
            current_first_banki = current['banki.ru'][0]['url']
            if current_first_banki != state['banki_first_url']:
                for review in current['banki.ru']:
                    new_reviews.append({
                        'text': review['text'],
                        'datetime_review': review['datetime_review'],
                        'source_id': review['source_id']
                    })
                    has_new = True

        # Проверка sravni.ru - если первый URL изменился, берем все текущие отзывы как новые
        if current['sravni.ru']:
            current_first_sravni = current['sravni.ru'][0]['url']
            if current_first_sravni != state['sravni_first_url']:
                for review in current['sravni.ru']:
                    new_reviews.append({
                        'text': review['text'],
                        'datetime_review': review['datetime_review'],
                        'source_id': review['source_id']
                    })
                    has_new = True

        if not has_new:
            return None

        new_banki_first = current['banki.ru'][0]['url'] if current['banki.ru'] else state['banki_first_url']
        new_sravni_first = current['sravni.ru'][0]['url'] if current['sravni.ru'] else state['sravni_first_url']
        self.save_state(new_banki_first, new_sravni_first)

        result = {
            'reviews': new_reviews,
            'total_new': len(new_reviews),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return result


if __name__ =="__main__":
    parser = ReviewMonitor()
    result = parser.get_new_reviews()
    print(result) if result else print("Новых отзывов не найдено!")



