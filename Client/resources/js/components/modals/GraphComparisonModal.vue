<template>
    <q-dialog v-model="model">
        <q-card style="width: 1500px; max-width: 1500vw; max-height: 500px; height: 100%">
            <q-card-section>
                <div class="text-h6">{{title}}</div>
            </q-card-section>

            <q-card-section class="q-pt-none">
                <base-graph style="width: 1400px" class="shadow-0" :tooltip="tooltip" type="category" :series="series"/>
            </q-card-section>

            <q-card-actions align="right">
                <q-btn flat label="OK" color="primary" v-close-popup />
            </q-card-actions>
        </q-card>
    </q-dialog>
</template>

<script setup>
import {watch} from "vue";
import BaseGraph from "../graphs/BaseGraph.vue";

const model = defineModel();
const props = defineProps(['title', 'product-list'])

//для графика
const tooltip = {
    trigger: 'axis'
}

const series = [
    {
        name: 'Email',
        type: 'line',
        stack: 'Total',
        data: [120, 132, 101, 134, 90, 230, 210]
    },
    {
        name: 'Union Ads',
        type: 'line',
        stack: 'Total',
        data: [220, 182, 191, 234, 290, 330, 310]
    },
    {
        name: 'Video Ads',
        type: 'line',
        stack: 'Total',
        data: [150, 232, 201, 154, 190, 330, 410]
    },
    {
        name: 'Direct',
        type: 'line',
        stack: 'Total',
        data: [320, 332, 301, 334, 390, 330, 320]
    },
    {
        name: 'Search Engine',
        type: 'line',
        stack: 'Total',
        data: [820, 932, 901, 934, 1290, 1330, 1320]
    }
]

watch(model, (newVal) => {
   if (newVal) {
   }
});
</script>

<style scoped>

</style>
