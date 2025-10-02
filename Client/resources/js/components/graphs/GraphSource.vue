<template>
    <base-loader v-if="loading"/>
    <base-graph v-if="series.length > 0" title-card="Региональная статистика по источнику" :column="column" type="category" :series="series" :legend="{}"/>
    <MessageNoData v-if="!loading && series.length === 0"/>
</template>

<script setup>
import BaseGraph from "./BaseGraph.vue";
import {StatisticApi} from "../../providers/StatisticApi.js";
import {onMounted, ref} from "vue";
import {useSelectDateStore} from "../../store/SelectDate.js";
import {useWatchRegion, useWatchStartDate, useWatchEndDate} from "../../composables/watchChanges.js";
import {useRegionStore} from "../../store/SelectRegion.js";
import BaseLoader from "../ui/BaseLoader.vue";
import MessageNoData from "../ui/MessageNoData.vue";

const storeRegion = useRegionStore();
const storeDate = useSelectDateStore();
const api = new StatisticApi();
const column = ['Положительно', 'Нейтрально', 'Отрицательно'];
const series = ref([]);
const loading = ref(true);

async function getData() {
    loading.value = true;
    series.value = [];
    try {
        let response = await api.getSourceStatistic(storeDate.startDate, storeDate.endDate, storeRegion.region.value);
        let sources = {data: []};
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
            sources.data.push(element.source);
            loading.value = false;
            console.log(loading.value)
        });
    } catch (e) {
        return e;
    } finally {
        loading.value = false;
    }
}

useWatchRegion(storeRegion, getData)
useWatchStartDate(storeDate, getData)
useWatchEndDate(storeDate, getData)

onMounted(async () => {
    await getData();
});
</script>
