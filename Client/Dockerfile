# Переместите копирование файлов ближе к концу для лучшего кэширования
FROM php:8.4-apache

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libpng-dev \
    libonig-dev \
    libxml2-dev \
    zip \
    unzip \
    libzip-dev \
    nodejs \
    npm \
    && docker-php-ext-install pdo_mysql mbstring exif pcntl bcmath gd zip \
    && a2enmod rewrite headers

# Установка Node.js (исправленная версия)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Установка Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Настройка рабочей директории
WORKDIR /var/www/html

# Копирование Apache конфига
COPY apache/000-default.conf /etc/apache2/sites-available/000-default.conf

# Включение mod_rewrite (уже включено выше, можно удалить дубликат)
# RUN a2enmod rewrite

# Копирование только файлов зависимостей сначала для кэширования
COPY src/composer.json src/composer.lock ./
COPY src/package.json src/package-lock.json* ./

# Установка зависимостей PHP
RUN composer install --no-dev --optimize-autoloader --no-scripts

# Установка зависимостей Node.js
RUN npm install

# Копирование остальных файлов проекта
COPY src/ .

# Установка прав
RUN chown -R www-data:www-data /var/www/html \
    && chmod -R 755 /var/www/html/storage \
    && chmod -R 755 /var/www/html/bootstrap/cache

# Сборка фронтенда
RUN npm run build

CMD ["apache2-foreground"]
