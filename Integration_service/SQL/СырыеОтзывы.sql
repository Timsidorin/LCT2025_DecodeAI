CREATE TABLE "public"."raw_reviews" (
  "uuid" uuid NOT NULL DEFAULT gen_random_uuid(),
  "source" varchar(100) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'API'::character varying,
  "text" text COLLATE "pg_catalog"."default" NOT NULL,
  "gender" varchar(10) COLLATE "pg_catalog"."default",
  "city" varchar(100) COLLATE "pg_catalog"."default",
  "region" varchar(100) COLLATE "pg_catalog"."default",
  "region_code" varchar(10) COLLATE "pg_catalog"."default",
  "datetime_review" timestamptz(6) NOT NULL,
  "created_at" timestamptz(6) NOT NULL DEFAULT now(),
  CONSTRAINT "raw_reviews_pkey" PRIMARY KEY ("uuid")
);


COMMENT ON TABLE "public"."raw_reviews" IS 'Таблица сырых (необработанных) отзывов';
COMMENT ON COLUMN "public"."raw_reviews"."uuid" IS 'Уникальный идентификатор отзыва';
COMMENT ON COLUMN "public"."raw_reviews"."source" IS 'Источник отзыва (banki.ru, sravni.ru, API)';
COMMENT ON COLUMN "public"."raw_reviews"."text" IS 'Текст отзыва';
COMMENT ON COLUMN "public"."raw_reviews"."gender" IS 'Пол автора: М (мужчина) или Ж (женщина)';
COMMENT ON COLUMN "public"."raw_reviews"."city" IS 'Город автора отзыва';
COMMENT ON COLUMN "public"."raw_reviews"."region" IS 'Регион';
COMMENT ON COLUMN "public"."raw_reviews"."region_code" IS 'Код региона (ISO формат, например: RU-MOW)';
COMMENT ON COLUMN "public"."raw_reviews"."datetime_review" IS 'Дата и время написания отзыва';
COMMENT ON COLUMN "public"."raw_reviews"."created_at" IS 'Время добавления в систему';

-- Создание индексов для ускорения поиска
CREATE INDEX "idx_raw_reviews_source" ON "public"."raw_reviews" ("source");
CREATE INDEX "idx_raw_reviews_gender" ON "public"."raw_reviews" ("gender");
CREATE INDEX "idx_raw_reviews_city" ON "public"."raw_reviews" ("city");
CREATE INDEX "idx_raw_reviews_region_code" ON "public"."raw_reviews" ("region_code");
CREATE INDEX "idx_raw_reviews_datetime_review" ON "public"."raw_reviews" ("datetime_review" DESC);
CREATE INDEX "idx_raw_reviews_created_at" ON "public"."raw_reviews" ("created_at" DESC);
