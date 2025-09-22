CREATE TABLE processed_reviews (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(100) NOT NULL DEFAULT 'API',
    text TEXT NOT NULL,
    rating VARCHAR(20) NULL,
    product VARCHAR(255) NULL,
    gender VARCHAR(10) NULL,
    city VARCHAR(100) NULL,
    region VARCHAR(100) NULL,
    region_code VARCHAR(10) NULL,
    datetime_review TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Индексы для оптимизации
CREATE INDEX idx_reviews_source ON processed_reviews(source);
CREATE INDEX idx_reviews_rating ON processed_reviews(rating);
CREATE INDEX idx_reviews_product ON processed_reviews USING gin(to_tsvector('russian', product));
CREATE INDEX idx_reviews_city ON processed_reviews(city);
CREATE INDEX idx_reviews_region ON processed_reviews(region);
CREATE INDEX idx_reviews_region_code ON processed_reviews(region_code);
CREATE INDEX idx_reviews_gender ON processed_reviews(gender);
CREATE INDEX idx_reviews_datetime_review ON processed_reviews(datetime_review);
CREATE INDEX idx_reviews_created_at ON processed_reviews(created_at);
CREATE INDEX idx_reviews_text ON processed_reviews USING gin(to_tsvector('russian', text));

-- Комментарии
COMMENT ON TABLE processed_reviews IS 'Таблица отзывов';
COMMENT ON COLUMN processed_reviews.source IS 'Источник отзыва (по умолчанию API)';
COMMENT ON COLUMN processed_reviews.text IS 'Текст отзыва';
COMMENT ON COLUMN processed_reviews.rating IS 'Тональность отзыва';
COMMENT ON COLUMN processed_reviews.product IS 'Название продукта';
COMMENT ON COLUMN processed_reviews.gender IS 'Пол автора';
COMMENT ON COLUMN processed_reviews.city IS 'Город автора';
COMMENT ON COLUMN processed_reviews.region IS 'Регион автора';
COMMENT ON COLUMN processed_reviews.region_code IS 'Код региона (автоматически)';
