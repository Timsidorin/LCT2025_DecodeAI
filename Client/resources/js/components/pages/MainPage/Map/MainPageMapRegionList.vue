<template>
    <q-select hint="Фильтр по региону" filled v-model="region" :options="list"/>
</template>

<script setup>
import {RegionApi} from "../../../../providers/RegionApi.js";
import {computed, onMounted, ref, watch} from "vue";
import {useRegionStore} from "../../../../store/MapSelectRegion.js";

const store = useRegionStore();
const api = new RegionApi();
const list = ref([]);

const region = computed({
    get: () => store.region,
    set: (value) => store.setRegion(value)
});

async function getData() {
    try {
        let response = await api.getListRegion();
        list.value = response.data.regions
    } catch (e) {
        return e;
    }
}

onMounted(async () => {
    await getData();
});
</script>

