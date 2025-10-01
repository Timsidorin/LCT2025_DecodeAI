<template>
    <q-card style="border-radius: 10px">
        <q-card-section>
            <div class="text-h6">{{titleCard}}</div>
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
const props = defineProps(['type', 'column', 'series', 'legend', 'titleCard', 'dataset', 'tooltip', 'matrix']);

function initGraph() {
    let chart = echarts.init(htmlElement.value);
    let option = {
        matrix: props.matrix,
        dataset: props.dataset,
        tooltip: {...props.tooltip},
        legend: {
            ...props.legend,
        },
        xAxis: {
            type: props.type,
            data: props.column,
        },
        yAxis: {
            type: 'value'
        },
        toolbox: {
            show: true,
            feature: {
                saveAsImage: {title: 'Скачать файл'}
            }
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
