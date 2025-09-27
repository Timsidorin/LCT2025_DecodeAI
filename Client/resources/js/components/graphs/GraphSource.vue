<template>
    <base-graph v-if="series.length > 0" :column="column" type="category" :series="series" :legend="legend"/>
</template>

<script setup>
import BaseGraph from "./BaseGraph.vue";
import {StatisticApi} from "../../providers/StatisticApi.js";
import {onMounted, ref} from "vue";
import {useSelectDateStore} from "../../store/MapSelectDate.js";

const store = useSelectDateStore();
const api = new StatisticApi();
const column = ['Положительно', 'Нейтрально', 'Отрицательно'];
const series = ref([]);
const legend = ref({});

async function getData() {
    try {
        let response = await api.getSourceStatistic(store.startDate, store.endDate);
        let sources = {data: []}
        response.data.sources.sources.forEach((element) => {
            series.value.push({
                data: [
                    element.positive_reviews,
                    element.neutral_reviews,
                    element.negative_reviews
                ],
                type: 'bar',
                name: element.source
            });
            sources.data.push(element.source)
        });
    } catch (e) {
        return e;
    }
}

onMounted(async () => {
    await getData();

})
</script>
