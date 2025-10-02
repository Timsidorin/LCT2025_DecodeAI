<template>
    <q-dialog v-model="model">
        <q-card style="width: 1500px; max-width: 1500vw; max-height: 500px; height: 100%">
            <q-card-section>
                <div class="text-h6">{{ title }}</div>
            </q-card-section>

            <q-card-section class="q-pt-none">
                <base-graph
                    style="width: 1400px"
                    class="shadow-0"
                    :tooltip="tooltip"
                    type="category"
                    :series="series"
                    :x-axis="xAxis"
                    :y-axis="yAxis"
                />
            </q-card-section>

            <q-card-actions align="right">
                <q-btn flat label="OK" color="primary" v-close-popup />
            </q-card-actions>
        </q-card>
    </q-dialog>
</template>

<script setup>
import { watch, computed } from "vue";
import BaseGraph from "../graphs/BaseGraph.vue";

const model = defineModel();
const props = defineProps(['title', 'product-list']);

// Для графика
const tooltip = {
    trigger: 'axis'
};

// Преобразуем данные в нужный формат
const series = computed(() => {
    if (!props.productList?.chart_data?.length) return [];

    const chartData = props.productList.chart_data;

    // Первая строка - заголовки
    const headers = chartData[0];

    // Создаем серии для каждого столбца данных (кроме первого - Month)
    const seriesData = [];

    for (let i = 1; i < headers.length; i++) {
        const data = [];

        // Проходим по всем строкам данных (начиная с 1, т.к. 0 - заголовки)
        for (let j = 1; j < chartData.length; j++) {
            data.push(chartData[j][i]);
        }

        seriesData.push({
            name: headers[i],
            type: 'line', // или 'bar' в зависимости от нужного типа
            data: data
        });
    }

    return seriesData;
});

// Настройки осей
const xAxis = computed(() => {
    if (!props.productList?.chart_data?.length) return { type: 'category' };

    const chartData = props.productList.chart_data;
    const categories = [];

    // Извлекаем месяцы из данных (начиная с 1, т.к. 0 - заголовки)
    for (let i = 1; i < chartData.length; i++) {
        categories.push(chartData[i][0]);
    }

    return {
        type: 'category',
        data: categories
    };
});

const yAxis = {
    type: 'value'
};

watch(model, (newVal) => {
    if (newVal) {
        console.log('Диалог открыт, данные:', props.productList);
    }
});
</script>

<style scoped>
</style>
