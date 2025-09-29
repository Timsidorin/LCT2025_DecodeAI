<template>
    <BaseLoader v-if="loading"/>
    <q-table
        v-else-if="row.length > 0"
        :rows="row"
        :columns="columns"
        row-key="id"
        virtual-scroll
        title="Региональная статистика по продуктам"
        style="height: 400px; border-radius: 10px; width: 100%"
    >
        <template v-slot:body="props">
            <q-tr :props="props">
                <q-td key="product" :props="props">
                    {{ props.row.product }}
                </q-td>
                <q-td key="positive_reviews" :props="props">
                    <q-badge color="green">
                        {{ props.row.positive_percentage }}%
                    </q-badge>
                    {{ props.row.positive_reviews }}
                </q-td>
                <q-td key="neutral_reviews" :props="props">
                    <q-badge color="amber">
                        {{ props.row.neutral_percentage }}%
                    </q-badge>
                    {{ props.row.neutral_reviews }}
                </q-td>
                <q-td key="negative_reviews" :props="props">
                    <q-badge color="red">
                        {{ props.row.negative_percentage }}%
                    </q-badge>
                    {{ props.row.negative_reviews }}
                </q-td>
            </q-tr>
        </template>
    </q-table>
</template>

<script setup>
import {useSelectDateStore} from "../../../store/SelectDate.js";
import {onMounted, ref} from "vue";
import {StatisticApi} from "../../../providers/StatisticApi.js";
import {useRegionStore} from "../../../store/SelectRegion.js";
import {useWatchRegion, useWatchStartDate, useWatchEndDate} from "../../../composables/watchChanges.js";
import BaseLoader from "../../ui/BaseLoader.vue";

const api = new StatisticApi();

const loading = ref(true);

const storeRegion = useRegionStore();
const storeDate = useSelectDateStore();

const row = ref([]);
const columns = [
    {name: 'product', field: 'product', align: 'center', label: 'Продукт', sortable: true,},
    {name: 'positive_reviews', field: 'positive_reviews', align: 'center', label: 'Положительно', sortable: true},
    {name: 'neutral_reviews', field: 'neutral_reviews', align: 'center', label: 'Нейтрально', sortable: true},
    {name: 'negative_reviews', field: 'negative_reviews', align: 'center', label: 'Отрицательно', sortable: true},
];

async function getDataTable() {
    loading.value = true;
    row.value = [];
    try {
        let result = await api.getTableStatistic(storeDate.startDate, storeDate.endDate, storeRegion.region?.value);
        row.value = result.data.regions_products;
        loading.value = false;
    } catch (e) {
        return e;
    }
}

useWatchRegion(storeRegion, getDataTable);
useWatchStartDate(storeDate, getDataTable);
useWatchEndDate(storeDate, getDataTable);

onMounted(async () => {
    await getDataTable();
});
</script>
