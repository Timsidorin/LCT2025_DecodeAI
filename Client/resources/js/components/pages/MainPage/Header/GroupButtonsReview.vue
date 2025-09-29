<template>
    <hr/>
    <div class="row items-center q-ml-xl q-pa-md">
        <div class="col-auto row q-col-gutter-md">
            <div class="col-auto">
                <span class="text-subtitle1">Регион</span>
                <q-select dense filled outlined v-model="region" :options="regionList" style="min-width: 200px"/>
            </div>
            <div class="col-auto">
                <span class="text-subtitle1">Город</span>
                <q-select dense filled outlined v-model="city" :options="cityList" style="min-width: 200px"/>
            </div>
            <MainPageListProduct/>
            <ReviewPageDatePicker/>
        </div>
    </div>
</template>

<script setup>
import {useRegionStore} from "../../../../store/SelectRegion.js";
import {useCityStore} from "../../../../store/SelectCity.js";
import {RegionApi} from "../../../../providers/RegionApi.js";
import {computed, onMounted, ref, watch} from "vue";
import MainPageListProduct from "../MainPageListProduct.vue";
import ReviewPageDatePicker from "../Review/ReviewPageDatePicker.vue";

const storeRegion = useRegionStore();
const storeCity = useCityStore();

const regionApi = new RegionApi();

const regionList = ref([]);
const cityList = ref([]);

function getListCityFromLocalStorage() {
    return JSON.parse(localStorage.getItem('selectedRegion')) ?? null;
}

async function getRegionList() {
    try {
        let response = await regionApi.getListRegion(true);
        regionList.value = response.data.region_hierarchy.regions;
    } catch (e) {
        return e;
    }
}

function addCityList(arr) {
    cityList.value = arr.cities.map((element) => {
        return {
            label: element.city_name,
            value: element.city_name
        }
    });
}

const region = computed({
    get: () => storeRegion.region,
    set: (value) => storeRegion.setRegion(value)
});
watch(region, (newValue) => {
    if (newValue) {
        addCityList(newValue);
    }
})

const city = computed({
    get: () => storeCity.city,
    set: (value) => storeCity.setCity(value)
});

onMounted( async () => {
    let listFromLocalStorage = getListCityFromLocalStorage();
    if (listFromLocalStorage) {
        addCityList(listFromLocalStorage);
    }
     await getRegionList();
})
</script>

<style scoped>
span {
    color: #4e4a4a;
}
</style>
