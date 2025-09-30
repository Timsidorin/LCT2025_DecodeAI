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

# Установка Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Установка Node.js (если нужна более новая версия)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Настройка рабочей директории
WORKDIR /var/www/html

# Копирование Apache конфига
COPY apache/000-default.conf /etc/apache2/sites-available/000-default.conf

# Включение mod_rewrite
RUN a2enmod rewrite

# Копирование проекта
COPY src/ /var/www/html/

# Установка прав
RUN chown -R www-data:www-data /var/www/html \
    && chmod -R 755 /var/www/html/storage \
    && chmod -R 755 /var/www/html/bootstrap/cache

# Установка зависимостей PHP
RUN composer install --no-dev --optimize-autoloader

# Установка зависимостей Node.js и сборка фронтенда
RUN npm install && npm run build

CMD ["apache2-foreground"]
