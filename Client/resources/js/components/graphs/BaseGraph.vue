<template>
    <q-card style="border-radius: 10px">
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

onMounted(() => {
    let chart = echarts.init(htmlElement.value);
    let option = {
        tooltip: {},
        legend: props.legend,
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
})
</script>

<style scoped>

</style>
