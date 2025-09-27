CREATE TABLE "public"."raw_reviews" (
  "uuid" uuid NOT NULL DEFAULT gen_random_uuid(),
  "source" varchar(100) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'API'::character varying,
  "text" text COLLATE "pg_catalog"."default" NOT NULL,
  "city" varchar(100) COLLATE "pg_catalog"."default",
  "region" varchar(100) COLLATE "pg_catalog"."default",
  "region_code" varchar(10) COLLATE "pg_catalog"."default",
  "datetime_review" timestamptz(6) NOT NULL,
  "created_at" timestamptz(6) NOT NULL DEFAULT now()
)