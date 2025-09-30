<template>
    <q-card>
        <q-card-section>
            <h6 class="q-my-none">Динамика отзывов по продукту</h6>
        </q-card-section>
        <q-card-section>
            <div ref="chartContainer" :style="{ width: '100%', height: chartHeight }"></div>
        </q-card-section>
    </q-card>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts';
import { StatisticApi } from "../../providers/StatisticApi.js";

const api = new StatisticApi();
const data = ref(null);

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
        default: '600px'
    }
});

async function getMatrixData() {
    try {
        const response = await api.getDataMatrix();
        data.value = response.data; // Предполагается, что response.data — это двумерный массив или объект с данными
    } catch (e) {
        console.error('Ошибка загрузки данных:', e);
    }
}

// Refs
const chartContainer = ref(null);
let myChart = null;

// Chart data
const generateSeries = () => {
    if (!data.value?.matrix_data) {
        return [];
    }

    const series = [];
    const matrixData = data.value.matrix_data;
    const yCategories = data.value.y_axis; // Продукты: ["Вклад", "Дебетовая карта", ...]
    const xCategories = data.value.x_axis; // Месяцы: ["May 2025", ...]

    // Но! В matrix_data ключи — это названия продуктов, а внутри — месяцы в формате "2025-05"
    // А x_axis — человекочитаемые названия ("May 2025"), но в данных — "2025-05"
    // Нужно сопоставить: "May 2025" → "2025-05"

    // Создадим маппинг из человекочитаемого месяца в ISO-месяц
    const monthNameToKey = {
        "May 2025": "2025-05",
        "Jun 2025": "2025-06",
        "Jul 2025": "2025-07",
        "Aug 2025": "2025-08",
        "Sep 2025": "2025-09"
        // Добавьте больше, если нужно
    };

    yCategories.forEach(product => {
        xCategories.forEach(monthName => {
            const monthKey = monthNameToKey[monthName];
            if (!monthKey) return;

            const cell = matrixData[product]?.[monthKey];
            if (!cell) return; // null или undefined — пропускаем

            const dataItems = [];

            // Позитивные
            if (cell.positive_pct > 0) {
                dataItems.push({
                    value: cell.positive_pct,
                    name: 'Позитивные',
                    itemStyle: { color: '#5cb85c' }
                });
            }

            // Негативные
            if (cell.negative_pct > 0) {
                dataItems.push({
                    value: cell.negative_pct,
                    name: 'Негативные',
                    itemStyle: { color: '#d9534f' }
                });
            }

            // Если оба 0 — всё равно рисуем хотя бы один сегмент?
            // Но в ваших данных такого нет, так что пропустим пустые
            if (dataItems.length === 0) return;

            series.push({
                type: 'pie',
                coordinateSystem: 'matrix',
                center: [monthName, product], // именно так: [x, y]
                radius: 15,
                data: dataItems,
                label: {
                    show: false
                },
                emphasis: {
                    label: {
                        show: true,
                        formatter: `${cell.total_reviews} отз.`
                    }
                }
            });
        });
    });

    return series;
};

const chartOption = ref({
    legend: {
        show: true,
        bottom: 40,
        data: ['Позитивные', 'Негативные']
    },
    matrix: {
        x: {
            data: [] // будет заполнено после загрузки
        },
        y: {
            data: [] // будет заполнено после загрузки
        },
        top: 80,
        bottom: 80
    },
    series: [],
    tooltip: {
        show: true
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

    // Обновляем оси
    chartOption.value.matrix.x.data = data.value.x_axis;
    chartOption.value.matrix.y.data = data.value.y_axis;

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
    updateChartData(); // теперь оси и данные подтянуты
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
</style>
