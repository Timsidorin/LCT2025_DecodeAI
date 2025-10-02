<template>
    <q-card style="border-radius: 10px">
        <q-card-section>
            <div class="text-h6">Сводка по отзывам</div>
        </q-card-section>
        <q-card-section>
            <div>
                <div ref="chartDom" style="width: 600px; height: 500px;"></div>
            </div>
        </q-card-section>
    </q-card>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
    chartData: {
        type: Array,
        default: () => [
            {value: 1048, name: 'Положительные', itemStyle: {color: '#2ea81d'}},
            {value: 735, name: 'Нейтральные', itemStyle: {color: '#cc0909'}},
            {value: 580, name: 'Отрицательные', itemStyle: {color: '#ffde00'}},
        ]
    },
});

const chartDom = ref(null);
const myChart = ref(null);

const initChart = () => {
    if (!chartDom.value) return;

    myChart.value = echarts.init(chartDom.value);

    const option = {
        tooltip: {
            trigger: 'item'
        },
        legend: {},
        series: [
            {
                type: 'pie',
                radius: '50%',
                data: props.chartData,
            }
        ]
    };

    myChart.value.setOption(option);
};

const handleResize = () => {
    if (myChart.value) {
        myChart.value.resize();
    }
};

// Наблюдатели за изменениями данных
watch(() => props.chartData, (newData) => {
    if (myChart.value) {
        myChart.value.setOption({
            series: [{
                data: newData
            }]
        });
    }
}, { deep: true });

// Жизненный цикл
onMounted(() => {
    initChart();
    window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
    if (myChart.value) {
        myChart.value.dispose();
    }
    window.removeEventListener('resize', handleResize);
});
</script>
