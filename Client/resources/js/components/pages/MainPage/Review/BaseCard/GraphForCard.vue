<template>
    <div ref="chartContainer" style="width: 300px; height: 300px;"></div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue';
import * as echarts from 'echarts';

const chartContainer = ref(null);
let myChart = null;

const initChart = () => {
    if (chartContainer.value && !myChart) {
        myChart = echarts.init(chartContainer.value);
        const option = {
            xAxis: {
                type: 'category',
                data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            },
            yAxis: {
                type: 'value'
            },
            series: [
                {
                    data: [820, 932, 901, 934, 1290, 1330, 1320],
                    type: 'line',
                    smooth: true
                }
            ]
        };

        myChart.setOption(option);
    }
};

onMounted(() => {
    initChart();
});

onUnmounted(() => {
    if (myChart) {
        myChart.dispose();
        myChart = null;
    }
});
</script>
