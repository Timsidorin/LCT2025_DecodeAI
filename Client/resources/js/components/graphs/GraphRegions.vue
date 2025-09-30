<template>
    <q-card>
        <q-card-section>
            <div ref="chartContainer" style="width: 100%; height: 400px;"></div>
        </q-card-section>
    </q-card>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import * as echarts from 'echarts';

const chartContainer = ref(null);
let myChart = null;

// Используем Unicode символы которые соответствуют Quasar иконкам
const weatherIcons = {
    Sunny: '☀️',
    Cloudy: '☁️',
    Showers: '🌧️'
};

const seriesLabel = {
    show: true
};

const option = {
    title: {
        text: 'Weather Statistics'
    },
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    },
    legend: {
        data: ['City Alpha', 'City Beta', 'City Gamma']
    },
    grid: {
        left: 100
    },
    toolbox: {
        show: true,
        feature: {
            saveAsImage: {}
        }
    },
    xAxis: {
        type: 'value',
        name: 'Days',
        axisLabel: {
            formatter: '{value}'
        }
    },
    yAxis: {
        type: 'category',
        inverse: true,
        data: ['Sunny', 'Cloudy', 'Showers'],
        axisLabel: {
            formatter: function (value) {
                return `${weatherIcons[value]}\n${value}`;
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
    series: [
        {
            name: 'City Alpha',
            type: 'bar',
            data: [165, 170, 30],
            label: seriesLabel,
            markPoint: {
                symbolSize: 1,
                symbolOffset: [0, '50%'],
                label: {
                    formatter: '{a|{a}\n}{b|{b} }{c|{c}}',
                    backgroundColor: 'rgb(242,242,242)',
                    borderColor: '#aaa',
                    borderWidth: 1,
                    borderRadius: 4,
                    padding: [4, 10],
                    lineHeight: 26,
                    position: 'right',
                    distance: 20,
                    rich: {
                        a: {
                            align: 'center',
                            color: '#fff',
                            fontSize: 18,
                            textShadowBlur: 2,
                            textShadowColor: '#000',
                            textShadowOffsetX: 0,
                            textShadowOffsetY: 1,
                            textBorderColor: '#333',
                            textBorderWidth: 2
                        },
                        b: {
                            color: '#333'
                        },
                        c: {
                            color: '#ff8811',
                            textBorderColor: '#000',
                            textBorderWidth: 1,
                            fontSize: 22
                        }
                    }
                },
                data: [
                    { type: 'max', name: 'max days: ' },
                    { type: 'min', name: 'min days: ' }
                ]
            }
        },
        {
            name: 'City Beta',
            type: 'bar',
            label: seriesLabel,
            data: [150, 105, 110]
        },
        {
            name: 'City Gamma',
            type: 'bar',
            label: seriesLabel,
            data: [220, 82, 63]
        }
    ]
};

const initChart = () => {
    if (!chartContainer.value) return;
    myChart = echarts.init(chartContainer.value);
    myChart.setOption(option);
};

const resizeChart = () => {
    myChart?.resize();
};

onMounted(() => {
    initChart();
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
