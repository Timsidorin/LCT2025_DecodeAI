<template>
    <q-card class="chart-card">
        <q-card-section>
            <h6 class="q-my-none">Динамика отзывов по продукту</h6>
        </q-card-section>

        <q-card-section class="chart-section">
            <div class="chart-wrapper">
                <div ref="chartContainer" :style="{ width: '100%', height: chartHeight }"></div>
            </div>
        </q-card-section>

    </q-card>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue';
import * as echarts from 'echarts';
import { StatisticApi } from "../../providers/StatisticApi.js";
import {useRegionStore} from "../../store/SelectRegion.js";
import {useWatchRegion} from "../../composables/watchChanges.js";
import {getFirstCharInUp} from "../../utils/mix.js";

const regionStore = useRegionStore();
const api = new StatisticApi();
const data = ref(null);
const loading = ref(false);

useWatchRegion(regionStore, getMatrixData)

// Props
const props = defineProps({
    xCnt: {
        type: Number,
        default: 9
    },
    yCnt: {
        type: Number,
        default: 6
    },
    chartHeight: {
        type: String,
        default: '500px'
    }
});

// Computed
const monthNames = computed(() => ({
    "January": "Январь",
    "February": "Февраль",
    "March": "Март",
    "April": "Апрель",
    "May": "Май",
    "June": "Июнь",
    "July": "Июль",
    "August": "Август",
    "September": "Сентябрь",
    "October": "Октябрь",
    "November": "Ноябрь",
    "December": "Декабрь"
}));

const productNames = computed(() => ({
    "Deposit": "Вклад",
    "Debit Card": "Дебетовая карта",
    "Credit": "Кредит",
    "Mortgage": "Ипотека",
    "Investment": "Инвестиции",
    "Insurance": "Страхование"
}));

async function getMatrixData() {
    loading.value = true;
    try {
        const response = await api.getDataMatrix(regionStore.region.value);
        data.value = response.data;
    } catch (e) {
        console.error('Ошибка загрузки данных:', e);
    } finally {
        loading.value = false;
    }
}

async function refreshData() {
    await getMatrixData();
}

// Refs
const chartContainer = ref(null);
let myChart = null;

// Methods
const translateMonth = (monthStr) => {
    if (!monthStr) return '';

    // Обрабатываем формат "May 2025"
    const [monthName, year] = monthStr.split(' ');
    const russianMonth = monthNames.value[monthName];

    if (russianMonth && year) {
        return `${russianMonth} ${year}`;
    }

    return monthStr;
};

const translateProduct = (productName) => {
    return getFirstCharInUp(productNames.value[productName] || productName);
};

const formatNumber = (num) => {
    return new Intl.NumberFormat('ru-RU').format(num);
};

// Chart data
const generateSeries = () => {
    if (!data.value?.matrix_data) {
        return [];
    }

    const series = [];
    const matrixData = data.value.matrix_data;
    const yCategories = data.value.y_axis.map(translateProduct);
    const xCategories = data.value.x_axis.map(translateMonth);

    // Создадим маппинг из человекочитаемого месяца в ISO-месяц
    const monthNameToKey = {
        "May 2025": "2025-05",
        "Jun 2025": "2025-06",
        "Jul 2025": "2025-07",
        "Aug 2025": "2025-08",
        "Sep 2025": "2025-09"
    };

    data.value.y_axis.forEach((originalProduct, yIndex) => {
        const product = translateProduct(originalProduct);

        data.value.x_axis.forEach((originalMonth, xIndex) => {
            const monthKey = monthNameToKey[originalMonth];
            if (!monthKey) return;

            const cell = matrixData[originalProduct]?.[monthKey];
            if (!cell) return;

            const dataItems = [];
            let totalValue = 0;

            // Позитивные
            if (cell.positive_pct > 0) {
                dataItems.push({
                    value: cell.positive_pct,
                    name: 'Позитивные',
                    itemStyle: {
                        color: '#28a745',
                        borderColor: '#fff',
                        borderWidth: 1
                    }
                });
                totalValue += cell.positive_pct;
            }

            // Негативные
            if (cell.negative_pct > 0) {
                dataItems.push({
                    value: cell.negative_pct,
                    name: 'Негативные',
                    itemStyle: {
                        color: '#dc3545',
                        borderColor: '#fff',
                        borderWidth: 1
                    }
                });
                totalValue += cell.negative_pct;
            }

            if (dataItems.length === 0) return;

            // Определяем размер круга в зависимости от общего количества отзывов
            const maxReviews = Math.max(...Object.values(matrixData[originalProduct] || {})
                .map(c => c?.total_reviews || 0));
            const radius = Math.max(10, Math.min(25, (cell.total_reviews / maxReviews) * 25));

            series.push({
                type: 'pie',
                coordinateSystem: 'matrix',
                center: [xIndex, yIndex],
                radius: radius,
                data: dataItems,
                label: {
                    show: false
                },
                emphasis: {
                    scale: true,
                    scaleSize: 5,
                    label: {
                        show: true,
                        backgroundColor: 'rgba(255,255,255,0.9)',
                        padding: [4, 8],
                        borderRadius: 4,
                        formatter: (params) => {
                            const positive = cell.positive_pct || 0;
                            const negative = cell.negative_pct || 0;
                            return `📊 ${cell.total_reviews} отз.\n👍 ${positive.toFixed(1)}%\n👎 ${negative.toFixed(1)}%`;
                        }
                    }
                },
                tooltip: {
                    formatter: (params) => {
                        const productName = product;
                        const monthName = xCategories[xIndex];
                        const positive = cell.positive_pct || 0;
                        const negative = cell.negative_pct || 0;
                        return `
                            <div style="font-weight: bold; margin-bottom: 8px;">
                                ${productName} - ${monthName}
                            </div>
                            <div>Всего отзывов: <b>${cell.total_reviews}</b></div>
                            <div style="color: #28a745;">Позитивные: ${positive.toFixed(1)}%</div>
                            <div style="color: #dc3545;">Негативные: ${negative.toFixed(1)}%</div>
                        `;
                    }
                }
            });
        });
    });

    return series;
};

const chartOption = ref({
    backgroundColor: '#fff',
    legend: {
        show: true,
        bottom: 10,
        icon: 'circle',
        itemWidth: 12,
        itemHeight: 12,
        textStyle: {
            fontSize: 12,
            fontWeight: 'normal'
        },
        data: [
            {
                name: 'Позитивные',
                itemStyle: { color: '#28a745' }
            },
            {
                name: 'Негативные',
                itemStyle: { color: '#dc3545' }
            }
        ]
    },
    matrix: {
        id: 'reviewMatrix',
        left: '3%',
        right: '3%',
        top: '10%',
        bottom: '15%',
        x: {
            data: [],
            axisLabel: {
                interval: 0,
                rotate: 45,
                margin: 10,
                fontSize: 11,
                fontWeight: 'normal',
                color: '#666'
            },
            axisLine: {
                lineStyle: {
                    color: '#e0e0e0'
                }
            },
            axisTick: {
                show: false
            }
        },
        y: {
            data: [],
            axisLabel: {
                interval: 0,
                fontSize: 11,
                fontWeight: 'normal',
                color: '#333',
                margin: 8,
                width: 120,
                overflow: 'truncate',
                ellipsis: '...'
            },
            axisLine: {
                lineStyle: {
                    color: '#e0e0e0'
                }
            },
            axisTick: {
                show: false
            }
        },
        itemStyle: {
            borderColor: '#fff',
            borderWidth: 1
        }
    },
    series: [],
    tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(255,255,255,0.95)',
        borderColor: '#e0e0e0',
        borderWidth: 1,
        textStyle: {
            color: '#333',
            fontSize: 12
        },
        extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 8px;'
    },
    grid: {
        show: false
    }
});

// Methods
const initChart = () => {
    if (!chartContainer.value) return;
    myChart = echarts.init(chartContainer.value);
    myChart.setOption(chartOption.value);
};

const resizeChart = () => {
    if (myChart) {
        myChart.resize();
    }
};

const updateChartData = () => {
    if (!data.value) return;

    // Обновляем оси с переведенными названиями
    chartOption.value.matrix.x.data = data.value.x_axis.map(translateMonth);
    chartOption.value.matrix.y.data = data.value.y_axis.map(translateProduct);

    // Обновляем серии
    chartOption.value.series = generateSeries();

    if (myChart) {
        myChart.setOption(chartOption.value, true);
    }
};

// Lifecycle
onMounted(async () => {
    await getMatrixData();
    initChart();
    updateChartData();
    window.addEventListener('resize', resizeChart);
});

onUnmounted(() => {
    if (myChart) {
        myChart.dispose();
        myChart = null;
    }
    window.removeEventListener('resize', resizeChart);
});

// Watchers
watch(() => [props.xCnt, props.yCnt, data.value], () => {
    updateChartData();
}, { deep: true });

</script>

<style scoped>
.chart-card {
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #e0e0e0;
    transition: box-shadow 0.3s ease;
}

.chart-card:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
}


.card-header h6 {
    font-size: 1.1rem;
    margin: 0;
}

.chart-section {
    padding: 16px;
    position: relative;
}

.chart-wrapper {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    background: #fafafa;
}

.card-footer {
    background-color: #f8f9fa;
    border-top: 1px solid #e9ecef;
    padding: 12px 20px;
}

.refresh-btn {
    color: #6c757d;
    transition: all 0.3s ease;
}

.refresh-btn:hover {
    color: #495057;
    transform: rotate(180deg);
}

:deep(.echarts-tooltip) {
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

:deep(.echarts-legend) {
    padding: 8px 0 !important;
}
</style>
