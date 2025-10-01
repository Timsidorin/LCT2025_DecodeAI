<template>
    <q-card class="custom-card custom-card--enhanced shadow-8">
        <q-card-section class="scroll-content modal-body">
            <div class="q-pa-md">
                <div class="filter-section">
                    <q-icon name="filter_alt" size="sm" class="q-mr-sm" />
                    <span class="text-subtitle1 text-weight-medium">Выберите продукты для сравнения</span>
                </div>

                <div class="products-grid">
                    <div
                        v-for="(product, index) in listProduct"
                        :key="product.value.value"
                        class="product-card-wrapper"
                    >
                        <div class="card" :style="{ animationDelay: `${index * 0.1}s` }">
                            <div class="card-header">
                                <h3 class="product-name">{{ product.label }}</h3>
                                <q-checkbox
                                    :model-value="selectedProduct.some(p => p.value === product.value.value)"
                                    @update:model-value="toggleProduct(product)"
                                    color="primary"
                                    class="product-checkbox"
                                />
                            </div>

                            <q-slide-transition>
                                <div v-if="selectedProduct.some(p => p.value === product.value.value)" class="card-body">
                                    <div class="type-options-section">
                                        <div class="section-title">
                                            <q-icon name="reviews" />
                                            <span>Типы отзывов</span>
                                        </div>

                                        <div class="sentiment-bars">
                                            <div
                                                v-for="type in typeCompOptions"
                                                :key="type.value"
                                                class="sentiment-bar type-option"
                                                :class="`type-option--${type.value}`"
                                            >
                                                <div class="sentiment-label">
                                                    <q-icon :name="getTypeIcon(type.value)" />
                                                    <span>{{ type.label }}</span>
                                                </div>
                                                <q-checkbox
                                                    :model-value="getSelectedType(product, type.value)"
                                                    @update:model-value="toggleType(product, type.value, $event)"
                                                    :color="getTypeColor(type.value)"
                                                    class="type-checkbox"
                                                />
                                            </div>
                                        </div>

                                        <div
                                            class="selection-summary-card"
                                            v-if="getSelectedTypesCount(product) > 0"
                                        >
                                            <q-icon name="check_circle" color="positive" />
                                            <span>Выбрано: {{ getSelectedTypesCount(product) }} из {{ typeCompOptions.length }}</span>
                                        </div>
                                    </div>
                                </div>
                            </q-slide-transition>
                        </div>
                    </div>
                </div>

                <div v-if="listProduct.length === 0" class="no-data">
                    <q-icon name="inventory_2" size="xl" color="grey-5" />
                    <h3>Продукты не найдены</h3>
                    <p>Нет доступных продуктов для анализа</p>
                </div>
            </div>
        </q-card-section>

        <q-card-actions class="modal-actions">
            <div class="selection-info">
                <q-icon name="checklist" />
                <span>Выбрано продуктов: {{ selectedProduct.length }}</span>
            </div>
            <q-btn
                :disable="disabledStatus"
                color="primary"
                label="Сравнить"
                @click="compare"
                class="compare-btn"
                :class="{ 'pulse-animation': !disabledStatus }"
                icon="compare_arrows"
            />
        </q-card-actions>
    </q-card>

    <graph-comparison-modal
        :product-list="dataPropsCompare"
        title="Сравнение по продуктам"
        v-model="statusModel"
    />
</template>

<script setup>
import GraphComparisonModal from "../modals/GraphComparisonModal.vue";
import {onMounted, ref, computed} from "vue";
import {ProductApi} from "../../providers/ProductApi.js";
import {StatisticApi} from "../../providers/StatisticApi.js";
import {useSelectDateStore} from "../../store/SelectDate.js";

const store = useSelectDateStore();
const api = new ProductApi();
const apiStatistic = new StatisticApi();
const listProduct = ref([]);

async function getListProduct() {
    try {
        let response = await api.getListProduct();
        listProduct.value = response.data.products_analysis.map((element) => {
            return {
                label: element.product,
                value: {value: element.product, type: []}
            }
        });
    } catch (e) {
        return e;
    }
}

onMounted(async () => {
    await getListProduct();
});

const selectedProduct = ref([]);
const statusModel = ref(false);
const typeCompOptions = [
    {
        label: 'Положительно',
        value: 'positive'
    },
    {
        label: 'Нейтрально',
        value: 'neutral'
    },
    {
        label: 'Отрицательно',
        value: 'negative'
    }
];

const dataPropsCompare = ref(null);

// Новые методы для управления выбором
function toggleProduct(product) {
    const index = selectedProduct.value.findIndex(p => p.value === product.value.value);

    if (index === -1) {
        // Добавляем продукт
        selectedProduct.value.push({
            value: product.value.value,
            type: []
        });
    } else {
        // Удаляем продукт
        selectedProduct.value.splice(index, 1);
    }
}

function toggleType(product, typeValue, isSelected) {
    const productIndex = selectedProduct.value.findIndex(p => p.value === product.value.value);

    if (productIndex === -1) return;

    if (isSelected) {
        // Добавляем тип, если его еще нет
        if (!selectedProduct.value[productIndex].type.includes(typeValue)) {
            selectedProduct.value[productIndex].type.push(typeValue);
        }
    } else {
        // Удаляем тип
        const typeIndex = selectedProduct.value[productIndex].type.indexOf(typeValue);
        if (typeIndex > -1) {
            selectedProduct.value[productIndex].type.splice(typeIndex, 1);
        }
    }
}

function getSelectedType(product, typeValue) {
    const selected = selectedProduct.value.find(p => p.value === product.value.value);
    return selected ? selected.type.includes(typeValue) : false;
}

function getSelectedTypesCount(product) {
    const selected = selectedProduct.value.find(p => p.value === product.value.value);
    return selected ? selected.type.length : 0;
}

function getTypeIcon(type) {
    const icons = {
        positive: 'sentiment_very_satisfied',
        neutral: 'sentiment_neutral',
        negative: 'sentiment_very_dissatisfied'
    };
    return icons[type] || 'help';
}

function getTypeColor(type) {
    const colors = {
        positive: 'positive',
        neutral: 'primary',
        negative: 'negative'
    };
    return colors[type] || 'primary';
}

function structureForApi(data) {
    let newArray = [];
    data.forEach((element) => {
        element.type.forEach((type) => {
            let obj = {
                name: element.value,
                type: type
            };
            newArray.push(obj)
        });
    });
    return newArray;
}

async function compare() {
    try {
        let newStruct = structureForApi(selectedProduct.value);
        let response = await apiStatistic.getCompareData({
            region_code: "RU-MOW",
            date_from: store.startDate,
            date_to: store.endDate,
            products: newStruct
        });
        dataPropsCompare.value = response.data;
        statusModel.value = !statusModel.value;
    } catch (e) {
        return e;
    }
}

const disabledStatus = computed(() => {
    return selectedProduct.value.length === 0 ||
        !selectedProduct.value.some(product => product.type.length > 0);
});
</script>

<style scoped>
.custom-card--enhanced {
    width: 100%;
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    height: 600px;
    overflow: hidden;
}

.card-header {
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
    width: 100%;
}

.title-section {
    flex: 1;
}

.modal-title {
    margin: 0 0 4px 0;
    color: #1a237e;
    font-weight: 600;
    font-size: 1.5rem;
}

.subtitle {
    color: #666;
    font-size: 0.9rem;
}

.header-icon {
    font-size: 2.5rem;
    opacity: 0.8;
    color: #1a237e;
}

.scroll-content {
    overflow: auto;
    flex-grow: 1;
    padding: 0;
}

.modal-body {
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
}

.filter-section {
    display: flex;
    align-items: center;
    padding: 16px 20px;
    background-color: rgba(192, 203, 211, 0.05);
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid rgba(211, 222, 232, 0.1);
}

.products-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 20px;
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
    height: fit-content;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0;
    padding: 0;
    border: none;
    background: none;
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

.product-checkbox {
    transform: scale(1.2);
}

.card-body {
    margin-top: 20px;
    color: #333;
}

.type-options-section {
    background: rgba(245, 247, 250, 0.7);
    border-radius: 12px;
    padding: 16px;
}

.section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    color: #666;
    font-weight: 500;
}

.sentiment-bars {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 16px;
}

.sentiment-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    transition: all 0.2s ease;
}

.sentiment-bar:hover {
    background: rgba(255, 255, 255, 0.8);
}

.sentiment-label {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    font-size: 14px;
}

.type-option--positive .sentiment-label {
    color: #2e7d32;
}

.type-option--neutral .sentiment-label {
    color: #616161;
}

.type-option--negative .sentiment-label {
    color: #c62828;
}

.type-checkbox {
    transform: scale(1.1);
}

.selection-summary-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    background: rgba(76, 175, 80, 0.1);
    border-radius: 8px;
    color: #2e7d32;
    font-size: 14px;
    font-weight: 500;
}

.no-data {
    text-align: center;
    padding: 60px 20px;
    color: #666;
}

.no-data h3 {
    margin: 16px 0 8px 0;
    color: #333;
}

.modal-actions {
    background: white;
    border-top: 1px solid #e0e0e0;
    padding: 16px 24px;
    justify-content: space-between;
    align-items: center;
}

.selection-info {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #666;
    font-weight: 500;
}

.compare-btn {
    border-radius: 24px;
    padding: 10px 28px;
    font-weight: 600;
    transition: all 0.3s ease;
    text-transform: none;
    font-size: 1rem;
}

.compare-btn:hover:not(.disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(25, 118, 210, 0.3);
}

.pulse-animation {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(25, 118, 210, 0.4);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(25, 118, 210, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(25, 118, 210, 0);
    }
}

/* Стилизация скроллбара */
.scroll-content::-webkit-scrollbar {
    width: 6px;
}

.scroll-content::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

.scroll-content::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 10px;
}

.scroll-content::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
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

    .header-content {
        flex-direction: column;
        gap: 12px;
    }

    .header-icon {
        align-self: flex-end;
    }

    .modal-actions {
        flex-direction: column;
        gap: 12px;
        align-items: stretch;
    }

    .compare-btn {
        width: 100%;
    }
}
</style>
