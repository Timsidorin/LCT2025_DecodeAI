<template>
    <q-dialog v-model="model" maximized>
        <q-card class="modal-card">
            <q-card-section class="modal-header">
                <div class="header-content">
                    <div class="title-section">
                        <h4 class="modal-title">
                            {{ gender === 'male' ? 'Топ продукты у мужчин' : 'Топ продукты у женщин' }}
                        </h4>
                        <div class="subtitle">Детальная аналитика по продуктам</div>
                    </div>
                    <q-btn
                        flat
                        round
                        icon="close"
                        class="close-btn"
                        @click="model = false"
                    />
                </div>
            </q-card-section>

            <q-card-section class="modal-body">
                <div class="products-grid">
                    <div
                        v-for="(product, index) in sortedProducts"
                        :key="product.product"
                        class="product-card-wrapper"
                    >
                        <div class="card" :style="{ animationDelay: `${index * 0.1}s` }">
                            <div class="card-header">
                                <h3 class="product-name">{{ getFirstCharInUp(product.product) }}</h3>
                                <q-badge
                                    :color="getRatingColor(product.avg_rating)"
                                    class="rating-badge"
                                    text-color="white"
                                    :label="product.avg_rating.toFixed(1)"
                                />
                            </div>

                            <div class="card-body">
                                <div class="reviews-info">
                                    <div class="reviews-count">
                                        <q-icon name="forum" />
                                        {{ formatNumber(product.total_reviews) }} отзывов
                                    </div>
                                </div>

                                <div class="sentiment-bars">
                                    <div class="sentiment-bar">
                                        <div class="sentiment-label">
                                            <q-icon name="thumb_up" />
                                            <span>Положительные</span>
                                        </div>
                                        <q-linear-progress
                                            :value="product.positive_ratio / 100"
                                            color="positive"
                                            track-color="grey-3"
                                            class="sentiment-progress"
                                        />
                                        <span class="sentiment-percent">{{ product.positive_ratio }}%</span>
                                    </div>

                                    <div class="sentiment-bar">
                                        <div class="sentiment-label">
                                            <q-icon name="thumb_down" />
                                            <span>Отрицательные</span>
                                        </div>
                                        <q-linear-progress
                                            :value="product.negative_ratio / 100"
                                            color="negative"
                                            track-color="grey-3"
                                            class="sentiment-progress"
                                        />
                                        <span class="sentiment-percent">{{ product.negative_ratio }}%</span>
                                    </div>
                                </div>

                                <div
                                    class="satisfaction-score"
                                    :style="{
                                        backgroundColor: getSatisfactionBgColor(product.satisfaction_score),
                                        color: getSatisfactionTextColor(product.satisfaction_score)
                                    }"
                                >
                                    <q-icon name="trending_up" />
                                    Удовлетворенность: {{ product.satisfaction_score }}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div v-if="!data || data.length === 0" class="no-data-modal">
                    <q-icon name="info" size="xl" color="grey-5" />
                    <h3>Нет данных для отображения</h3>
                    <p>Данные по продуктам для этой категории отсутствуют</p>
                </div>
            </q-card-section>

            <q-card-actions class="modal-actions">
                <q-btn
                    flat
                    color="primary"
                    icon="arrow_back"
                    label="Назад"
                    @click="model = false"
                    class="back-btn"
                />
            </q-card-actions>
        </q-card>
    </q-dialog>
</template>

<script setup>
import { computed } from 'vue';
import {getFirstCharInUp} from "../../utils/mix.js";

const props = defineProps(['data', 'gender']);
const model = defineModel();

// Сортируем продукты по рейтингу
const sortedProducts = computed(() => {
    if (!props.data || !Array.isArray(props.data)) {
        return [];
    }
    return [...props.data].sort((a, b) => b.avg_rating - a.avg_rating);
});

// Методы для вычисления цветов
const getRatingColor = (rating) => {
    if (rating >= 4) return 'positive';
    if (rating >= 3) return 'primary';
    if (rating >= 2) return 'warning';
    return 'negative';
};

const getSatisfactionBgColor = (score) => {
    if (score > 0) return 'rgba(76,175,80,0.2)';
    if (score < 0) return 'rgba(244,67,54,0.2)';
    return 'rgba(255,152,0,0.2)';
};

const getSatisfactionTextColor = (score) => {
    if (score > 0) return '#4caf50';
    if (score < 0) return '#f44336';
    return '#ff9800';
};

const formatNumber = (num) => {
    return num.toLocaleString('ru-RU');
};
</script>

<style scoped>
.modal-card {
    background: #f5f7fa;
    min-height: 100vh;
}

.modal-header {
    background: white;
    border-bottom: 1px solid #e0e0e0;
    padding: 20px 24px;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    max-width: 1200px;
    margin: 0 auto;
}

.title-section {
    flex: 1;
}

.modal-title {
    margin: 0 0 4px 0;
    color: #1a237e;
    font-weight: 600;
}

.subtitle {
    color: #666;
    font-size: 0.9rem;
}

.close-btn {
    color: #666;
}

.modal-body {
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
}

.products-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 20px;
    margin-bottom: 24px;
}

.product-card-wrapper {
    animation: slideUp 0.6s ease-out forwards;
    opacity: 0;
    transform: translateY(20px);
}

@keyframes slideUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
}

.product-name {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin: 0;
    flex: 1;
    margin-right: 12px;
    line-height: 1.3;
}

.rating-badge {
    min-width: 50px;
    font-size: 16px;
    font-weight: 600;
}

.card-body {
    color: #333;
}

.reviews-info {
    margin-bottom: 20px;
}

.reviews-count {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #666;
    font-size: 14px;
}

.sentiment-bars {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 20px;
}

.sentiment-bar {
    display: flex;
    align-items: center;
    gap: 12px;
}

.sentiment-label {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 120px;
    font-size: 14px;
    color: #666;
}

.sentiment-progress {
    flex: 1;
    height: 8px;
    border-radius: 4px;
}

.sentiment-percent {
    min-width: 40px;
    text-align: right;
    font-size: 14px;
    font-weight: 500;
    color: #333;
}

.satisfaction-score {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 500;
    margin-top: 8px;
}

.no-data-modal {
    text-align: center;
    padding: 60px 20px;
    color: #666;
}

.no-data-modal h3 {
    margin: 16px 0 8px 0;
    color: #333;
}

.modal-actions {
    background: white;
    border-top: 1px solid #e0e0e0;
    padding: 16px 24px;
    justify-content: center;
}

.back-btn {
    font-weight: 500;
    text-transform: none;
    border-radius: 8px;
}

/* Адаптивность */
@media (max-width: 768px) {
    .products-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }

    .modal-body {
        padding: 16px;
    }

    .card {
        padding: 20px;
    }

    .sentiment-bar {
        flex-direction: column;
        align-items: stretch;
        gap: 8px;
    }

    .sentiment-label {
        min-width: auto;
    }
}
</style>
