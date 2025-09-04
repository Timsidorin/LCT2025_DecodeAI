


# Конфигурация сайтов для парсинга
SITES_TO_PARSING = [
    {
        "name": "banki.ru",
        "url": "https://www.banki.ru/services/responses/bank/gazprombank/?type=all",
        "review_selector": "div[data-test='responses__response']",
        "config": {
            "id_attr": "data-response-id",
            "date_selector": "span.Responsesstyled__StyledItemSmallText-sc-150koqm-4",
            "text_selector": "div.Responsesstyled__StyledItemText-sc-150koqm-3 a",
            "title_selector": "a[data-test='link-text']",
            "rating_attr": "data-test-grade",
            "author_selector": "div.Responsesstyled__StyledItemInfo-sc-150koqm-2",
            "date_format": "%d.%m.%Y %H:%M"
        }
    },
    {
        "name": "sravni.ru",
        "url": "https://www.sravni.ru/bank/gazprombank/otzyvy/",
        "review_selector": "div.review-card_wrapper__gnPSK",
        "config": {
            "id_from_url": True,
            "date_selector": "div.h-color-D30",
            "text_selector": "div.review-card_text__jTUSq",
            "title_selector": "div.review-card_title__zYdxx",
            "rating_selector": "div[data-qa='Rate']",
            "author_selector": "div.h-color-D100",
            "date_format": "relative"
        }
    }
]