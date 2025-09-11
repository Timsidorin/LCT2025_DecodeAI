CREATE TABLE reviews (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(100) NOT NULL DEFAULT 'API',
    text TEXT NOT NULL,
    rating VARCHAR(20) NULL,
    product VARCHAR(255) NULL,
    gender VARCHAR(10) NULL,
    city VARCHAR(100) NULL,
    datetime_review TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Индексы для оптимизации
CREATE INDEX idx_reviews_source ON reviews(source);
CREATE INDEX idx_reviews_rating ON reviews(rating);
CREATE INDEX idx_reviews_product ON reviews USING gin(to_tsvector('russian', product));
CREATE INDEX idx_reviews_city ON reviews(city);
CREATE INDEX idx_reviews_gender ON reviews(gender);
CREATE INDEX idx_reviews_datetime_review ON reviews(datetime_review);
CREATE INDEX idx_reviews_created_at ON reviews(created_at);
CREATE INDEX idx_reviews_text ON reviews USING gin(to_tsvector('russian', text));

-- Комментарии
COMMENT ON TABLE reviews IS 'Таблица отзывов';
COMMENT ON COLUMN reviews.source IS 'Источник отзыва (по умолчанию API)';
COMMENT ON COLUMN reviews.text IS 'Текст отзыва';
COMMENT ON COLUMN reviews.rating IS 'Тональность отзыва';
COMMENT ON COLUMN reviews.product IS 'Название продукта';
COMMENT ON COLUMN reviews.gender IS 'Пол автора';
COMMENT ON COLUMN reviews.city IS 'Город автора';
