<template>
    <div class="row q-gutter-x-xl no-wrap">
        <TotalReviews :chart-data="totalReview" v-if="totalReview" class="col-4"/>
        <GrowthMetrics :data="review24Hours" v-if="review24Hours" class="col-4"/>
        <GraphComparison class="col-4"/>
    </div>
</template>

<script setup>
import TotalReviews from "./BasicSummary/TotalReviews.vue";
import {StatisticApi} from "../../providers/StatisticApi.js";
import {onMounted, ref} from "vue";
import GrowthMetrics from "./BasicSummary/GrowthMetrics.vue";
import GraphComparison from "./GraphComparison.vue";

const api = new StatisticApi();
const totalReview = ref(null);
const review24Hours = ref(null);

async function getBasicSummary() {
    try {
        let response = await api.getBasicSummary();
        totalReview.value = [
            {
                value: response.data.overview.sentiment_distribution.positive,
                name: 'Положительные',
                itemStyle: {color: 'rgb(109,216,109)'}
            },
            {
                value: response.data.overview.sentiment_distribution.negative,
                name: 'Отрицательные',
                itemStyle: {color: 'rgb(223 43 79)'}
            },
            {
                value: response.data.overview.sentiment_distribution.neutral,
                name: 'Нейтральные',
                itemStyle: {color: 'rgb(255 219 25)'}
            }
        ];
        review24Hours.value = {growth_metrics: response.data.overview.growth_metrics, gender_distribution: response.data.overview.gender_distribution};
    } catch (e) {
        return e;
    }
}

onMounted(async () => {
    await getBasicSummary();
});
</script>

