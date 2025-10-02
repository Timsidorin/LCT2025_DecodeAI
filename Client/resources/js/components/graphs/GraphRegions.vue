<template>
    <q-card>
        <q-card-section>
            <h6 class="q-ma-none">Топ-3 региона по количеству отзывов</h6>
        </q-card-section>
        <q-card-section>
            <div ref="chartContainer" style="width: 100%; height: 400px;"></div>
        </q-card-section>
    </q-card>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts';
import { MapApi } from "../../providers/MapApi.js";

const api = new MapApi();
const data = ref(null);

async function getData() {
    try {
        let response = await api.coloringMap('positive');
        data.value = response.data.regions;
    } catch (e) {
        console.error('Error fetching data:', e);
        return e;
    }
}

const chartContainer = ref(null);
let myChart = null;

const seriesLabel = {
    show: true
};

const initChart = () => {
    if (!chartContainer.value || !data.value) return;

    myChart = echarts.init(chartContainer.value);

    // Сортируем регионы по total_reviews (по убыванию) и берем топ-3
    const sortedRegions = [...data.value].sort((a, b) => b.total_reviews - a.total_reviews);
    const displayRegions = sortedRegions.slice(0, 3);
    const regionNames = displayRegions.map(region => region.region_name);


    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function (params) {
                let result = `<strong>${params[0].axisValue}</strong><br/>`;
                params.forEach(param => {
                    const region = displayRegions.find(r => r.region_name === param.seriesName);
                    const total = region ? region.total_reviews : 0;
                    const percentage = total > 0 ? ((param.value / total) * 100).toFixed(1) : 0;
                    result += `${param.marker} ${param.seriesName}: ${param.value} (${percentage}%)<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: regionNames
        },
        grid: {
            left: 100
        },
        toolbox: {
            show: true,
            feature: {
                saveAsImage: {title: 'Скачать график'}
            }
        },
        xAxis: {
            type: 'value',
            name: 'Количество отзывов',
            axisLabel: {
                formatter: '{value}'
            }
        },
        yAxis: {
            type: 'category',
            inverse: true,
            data: ['Положительные', 'Отрицательные', 'Нейтральные'],
            axisLabel: {
                formatter: function (value) {
                    const icons = {
                        'Положительные': '👍',
                        'Отрицательные': '👎',
                        'Нейтральные': '😐'
                    };
                    return `${icons[value]}\n${value}`;
                },
                margin: 20,
                rich: {
                    value: {
                        lineHeight: 30,
                        align: 'center'
                    }
                }
            }
        },
        series: displayRegions.map((region, index) => {
            const colors = ['#5470c6', '#91cc75', '#fac858']; // Цвета для разных регионов

            return {
                name: region.region_name,
                type: 'bar',
                data: [
                    region.positive_reviews,
                    region.negative_reviews,
                    region.neutral_reviews
                ],
                label: {
                    ...seriesLabel,
                    formatter: function(params) {
                        // Показываем проценты в подписях
                        const total = region.total_reviews;
                        const percentage = total > 0 ? ((params.value / total) * 100).toFixed(1) : 0;
                        return `${params.value}\n(${percentage}%)`;
                    }
                },
                itemStyle: {
                    color: colors[index]
                },
                markPoint: {
                    symbolSize: 1,
                    symbolOffset: [0, '50%'],

                    data: [
                        { type: 'max', name: 'макс: ' },
                        { type: 'min', name: 'мин: ' }
                    ]
                }
            };
        })
    };

    myChart.setOption(option);
};

const resizeChart = () => {
    myChart?.resize();
};

// Следим за изменением данных
watch(data, () => {
    if (data.value) {
        initChart();
    }
});

onMounted(async () => {
    await getData();
    window.addEventListener('resize', resizeChart);
});

onUnmounted(() => {
    if (myChart) {
        myChart.dispose();
        myChart = null;
    }
    window.removeEventListener('resize', resizeChart);
});
</script>
