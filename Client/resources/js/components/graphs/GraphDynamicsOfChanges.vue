<template>
    <q-card style="border-radius: 10px">
        <q-card-section>
            <div class="text-h6">Динамика отзывов по продукту</div>
        </q-card-section>
        <q-card-section>
            <div style="height: 300px" ref="productGraphDiv"/>
        </q-card-section>
    </q-card>
</template>

<script setup>
import { StatisticApi } from "../../providers/StatisticApi.js";
import { useSelectDateStore } from "../../store/SelectDate.js";
import { useProductStore } from "../../store/SelectProduct.js"
import { useWatchProduct, useWatchRegion } from "../../composables/watchChanges.js";
import { useRegionStore } from "../../store/SelectRegion.js";
import { onMounted, ref, onUnmounted, nextTick } from "vue";
import * as echarts from 'echarts';

const dateStore = useSelectDateStore();
const productStore = useProductStore();
const regionStore = useRegionStore();
const api = new StatisticApi();
const rawData = ref('');
const productGraphDiv = ref(null);
let myChart = null;

// Инициализация ECharts
function initChart() {
    if (productGraphDiv.value && !myChart) {
        myChart = echarts.init(productGraphDiv.value);

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
function run(_rawData) {
    if (!myChart) return;

    const option = {
        dataset: [
            {
                id: 'dataset_raw',
                source: _rawData
            }
        ],
        title: {
            text: 'Тренд отзывов по месяцам',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: ['Положительные отзывы', 'Негативные отзывы', 'Нейтральные отзывы'],
        },
        xAxis: {
            type: 'category',
            nameLocation: 'middle',
            name: 'Период'
        },
        yAxis: {
            name: 'Количество отзывов'
        },
        series: [
            {
                name: 'Положительные отзывы',
                type: 'line',
                showSymbol: false,
                smooth: true,
                lineStyle: {
                    color: '#52c41a',
                    width: 3
                },
                encode: {
                    x: 'Month',
                    y: 'Positive_Reviews',
                    itemName: 'Month',
                    tooltip: ['Positive_Reviews']
                }
            },
            {
                name: 'Негативные отзывы',
                type: 'line',
                showSymbol: false,
                smooth: true,
                lineStyle: {
                    color: '#ff4d4f',
                    width: 3
                },
                encode: {
                    x: 'Month',
                    y: 'Negative_Reviews',
                    itemName: 'Month',
                    tooltip: ['Negative_Reviews']
                }
            },
            {
                name: 'Нейтральные отзывы',
                type: 'line',
                showSymbol: false,
                smooth: true,
                lineStyle: {
                    color: '#faad14',
                    width: 2,
                    type: 'dashed'
                },
                encode: {
                    x: 'Month',
                    y: 'Neutral_Reviews',
                    itemName: 'Month',
                    tooltip: ['Neutral_Reviews']
                }
            }
        ]
    };

    myChart.setOption(option);
}

async function getData() {
    try {
        let response = await api.getDynamicsOfChanges(
            dateStore.startDate,
            dateStore.endDate,
            productStore.product,
            regionStore.region
        );
        rawData.value = response.data;

        // Отрисовываем график после получения данных
        if (rawData.value) {
            run(rawData.value);
        }
    } catch (e) {
        console.error('Ошибка при получении данных:', e);
        return e;
    }
}

// Следим за изменениями продукта и региона
useWatchProduct(productStore, getData);
useWatchRegion(regionStore, getData);

onMounted(async () => {
    // Инициализируем график после монтирования компонента
    await nextTick();
    initChart();

    // Загружаем начальные данные
    await getData();
});
</script>

<style scoped>

</style>
