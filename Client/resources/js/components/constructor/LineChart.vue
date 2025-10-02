<template>
    <q-card style="border-radius: 10px">
        <q-card-section>
            <div class="text-h6">{{ title }}</div>
        </q-card-section>
        <q-card-section>
            <div :style="{ height: height + 'px' }" ref="chartDiv"/>
        </q-card-section>
    </q-card>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from "vue";
import * as echarts from 'echarts';

// Пропсы компонента
const props = defineProps({
    // Заголовок графика
    title: {
        type: String,
        default: 'Динамика отзывов по продукту'
    },
    // Высота графика
    height: {
        type: Number,
        default: 500
    },
    // Данные для графика
    chartData: {
        type: Array,
        default: () => []
    },
    // Настройки серий
    seriesConfig: {
        type: Array,
        default: () => [
            {
                name: 'Положительные отзывы',
                dataKey: 'Positive_Reviews',
                color: '#52c41a',
                lineWidth: 3,
                smooth: true,
                showSymbol: false
            },
            {
                name: 'Негативные отзывы',
                dataKey: 'Negative_Reviews',
                color: '#ff4d4f',
                lineWidth: 3,
                smooth: true,
                showSymbol: false
            },
            {
                name: 'Нейтральные отзывы',
                dataKey: 'Neutral_Reviews',
                color: '#faad14',
                lineWidth: 2,
                lineType: 'dashed',
                smooth: true,
                showSymbol: false
            }
        ]
    },
    // Настройка оси X
    xAxisConfig: {
        type: Object,
        default: () => ({
            name: 'Период',
            dataKey: 'Month'
        })
    },
    // Настройка оси Y
    yAxisConfig: {
        type: Object,
        default: () => ({
            name: 'Количество отзывов'
        })
    },
    // Показывать ли легенду
    showLegend: {
        type: Boolean,
        default: true
    },
    // Включить tooltip
    showTooltip: {
        type: Boolean,
        default: true
    }
});

const chartDiv = ref(null);
let myChart = null;

// Инициализация ECharts
function initChart() {
    if (chartDiv.value && !myChart) {
        myChart = echarts.init(chartDiv.value);

        // Обработчик изменения размера окна
        const resizeHandler = () => {
            if (myChart) {
                myChart.resize();
            }
        };
        window.addEventListener('resize', resizeHandler);

        // Очистка при уничтожении компонента
        onUnmounted(() => {
            window.removeEventListener('resize', resizeHandler);
            if (myChart) {
                myChart.dispose();
                myChart = null;
            }
        });
    }
}

// Функция для отрисовки графика
function renderChart() {
    if (!myChart || !props.chartData.length) return;

    const option = {
        dataset: [
            {
                id: 'dataset_raw',
                source: props.chartData
            }
        ],
        tooltip: props.showTooltip ? {
            trigger: 'axis'
        } : undefined,
        legend: props.showLegend ? {
            data: props.seriesConfig.map(series => series.name)
        } : undefined,
        xAxis: {
            type: 'category',
            nameLocation: 'middle',
            name: props.xAxisConfig.name
        },
        yAxis: {
            name: props.yAxisConfig.name
        },
        series: props.seriesConfig.map(series => ({
            name: series.name,
            type: 'line',
            showSymbol: series.showSymbol !== undefined ? series.showSymbol : true,
            smooth: series.smooth !== undefined ? series.smooth : false,
            lineStyle: {
                color: series.color,
                width: series.lineWidth || 2,
                type: series.lineType || 'solid'
            },
            encode: {
                x: props.xAxisConfig.dataKey,
                y: series.dataKey,
                itemName: props.xAxisConfig.dataKey,
                tooltip: [series.dataKey]
            }
        }))
    };

    myChart.setOption(option);
}

// Наблюдаем за изменениями данных
watch(() => props.chartData, (newData) => {
    if (newData && newData.length > 0) {
        renderChart();
    }
}, { deep: true });

// Наблюдаем за изменениями конфигурации
watch(() => props.seriesConfig, () => {
    renderChart();
}, { deep: true });

onMounted(async () => {
    // Инициализируем график после монтирования компонента
    await nextTick();
    initChart();

    // Отрисовываем график если данные уже есть
    if (props.chartData && props.chartData.length > 0) {
        renderChart();
    }
});
</script>

<style scoped>
</style>
