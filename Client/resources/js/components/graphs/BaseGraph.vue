<template>
    <q-card style="border-radius: 10px">
        <q-card-section>
            <div class="text-h6">Региональная статистика по источнику</div>
        </q-card-section>
        <q-card-section>
            <div style="height: 300px" ref="baseGraphDiv"/>
        </q-card-section>
    </q-card>
</template>

<script setup>
import {onMounted, useTemplateRef} from 'vue';
import * as echarts from 'echarts';

const htmlElement = useTemplateRef('baseGraphDiv');
const props = defineProps(['type', 'column', 'series', 'legend']);

function initGraph() {
    let chart = echarts.init(htmlElement.value);
    let option = {
        tooltip: {},
        legend: {
            ...props.legend,
        },
        xAxis: {
            type: props.type,
            data: props.column
        },
        yAxis: {
            type: 'value'
        },
        series: props.series
    };
    option && chart.setOption(option);
}

onMounted(() => {
    initGraph();
})
</script>

<style scoped>

</style>
