<template>
    <div class="row">
        <div class="col-7">
            <main-page-map-russia />
            <graph-regions class="q-mt-md"/>
        </div>
        <div class="q-ml-xl column" style="width: 100%; max-width: 700px">
            <main-page-date-picker class="q-mb-md"/>
            <main-page-table class="q-mb-md"/>
            <div>
                <graph-source/>
            </div>
        </div>
    </div>
</template>

<script setup>
import MainPageMapRussia from "../../components/pages/MainPage/Map/MainPageMapRussia.vue";
import MainPageTable from "../../components/pages/MainPage/MainPageTable.vue";
import MainPageDatePicker from "../../components/pages/MainPage/MainPageDatePicker.vue";
import GraphSource from "../../components/graphs/GraphSource.vue";
import GraphRegions from "../../components/graphs/GraphRegions.vue";
</script>
