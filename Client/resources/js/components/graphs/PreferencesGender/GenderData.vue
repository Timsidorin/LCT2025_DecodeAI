<template>
    <q-card class="insight-card shadow-2">
        <q-card-section class="card-content">
            <div class="header-section">
                <div class="gender-icon" :class="gender">
                    <q-icon :name="gender === 'male' ? 'male' : 'female'" size="sm" />
                </div>
                <div class="title-section">
                    <h6 class="q-mt-none q-mb-xs card-title">{{ gender === 'male' ? 'Мужчины' : 'Женщины' }}</h6>
                    <div class="subtitle">Топ продукты</div>
                </div>
            </div>

            <div class="insights-section">
                <div v-for="(insight, index) in filteredInsights" :key="index" class="insight-item">
                    <div class="message-content" v-html="formatMessage(insight.message)"></div>
                    <div class="priority-indicator" :class="insight.priority"></div>
                </div>
                <div v-if="filteredInsights.length === 0" class="no-data">
                    <q-icon name="info" size="sm" />
                    <span>Нет данных для отображения</span>
                </div>
            </div>
        </q-card-section>

        <q-card-actions class="card-actions">
            <q-btn
                flat
                color="primary"
                icon="arrow_forward"
                label="Подробнее"
                class="details-btn"
                @click="modalStatus = !modalStatus"
            />
        </q-card-actions>
    </q-card>
    <gender-data-modal v-model="modalStatus" :data="data.top_products"/>
</template>

<script setup>
import { computed, ref } from 'vue';
import GenderDataModal from "../../modals/GenderDataModal.vue";

const props = defineProps(['data', 'title', 'insights', 'gender']);
const modalStatus = ref(false);
// Фильтруем insights в зависимости от гендера
const filteredInsights = computed(() => {
    if (!props.insights || !Array.isArray(props.insights)) {
        return [];
    }

    return props.insights.filter(insight => {
        if (props.gender === 'male') {
            return insight.type === 'male_top';
        } else if (props.gender === 'female') {
            return insight.type === 'female_top';
        }
        return false;
    });
});

// Форматируем сообщение, выделяя ключевое слово
const formatMessage = (message) => {
    if (!message) return '';

    // Убираем префикс "Топ продукт среди мужчин/женщин:"
    const cleanedMessage = message.replace(/^Топ продукт среди (мужчин|женщин):\s*/, '');

    // Разделяем на первое слово и остальной текст
    const words = cleanedMessage.split(' ');
    if (words.length === 0) return message;

    const firstWord = words[0];
    const restOfText = words.slice(1).join(' ');

    return `<strong class="highlighted-text">${firstWord}</strong>${restOfText ? ' ' + restOfText : ''}`;
};
</script>

<style scoped>
.insight-card {
    border-radius: 12px;
    transition: all 0.3s ease;
    overflow: hidden;
}


.card-content {
    padding: 20px;
}

.header-section {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.gender-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.gender-icon.male {
    background: linear-gradient(135deg, #2196F3, #1976D2);
    color: white;
}

.gender-icon.female {
    background: linear-gradient(135deg, #E91E63, #C2185B);
    color: white;
}

.title-section {
    flex: 1;
}

.card-title {
    font-weight: 600;
    color: #1a237e;
    font-size: 1.1rem;
}

.subtitle {
    font-size: 0.85rem;
    color: #666;
    opacity: 0.8;
}

.insights-section {
    margin: 16px 0;
}

.insight-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px;
    background: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 4px solid #2196F3;
}

.message-content {
    flex: 1;
    font-size: 0.95rem;
    line-height: 1.4;
    color: #333;
}

.priority-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}

.priority-indicator.info {
    background-color: #2196F3;
}

.priority-indicator.warning {
    background-color: #FF9800;
}

.priority-indicator.error {
    background-color: #F44336;
}

.no-data {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px;
    text-align: center;
    color: #666;
    background: #f5f5f5;
    border-radius: 8px;
    font-size: 0.9rem;
}

.card-actions {
    padding: 16px 20px 20px;
    border-top: 1px solid #f0f0f0;
}

.details-btn {
    font-weight: 500;
    text-transform: none;
    border-radius: 6px;
}

.highlighted-text {
    color: #1976d2;
    font-weight: 700;
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    padding: 4px 8px;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(33, 150, 243, 0.3);
}
</style>
