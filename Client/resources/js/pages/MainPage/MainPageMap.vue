<template>
    <div class="row">
        <div class="col-7">
            <main-page-map-russia />
        </div>
        <div class="q-ml-auto q-mr-xl column">
            <main-page-map-date-picker class="q-mb-md"/>
            <main-page-map-table class="q-mb-md"/>
            <graph-source/>
        </div>
    </div>
</template>

<script setup>
import MainPageMapRussia from "../../components/pages/MainPage/Map/MainPageMapRussia.vue";
import MainPageMapTable from "../../components/pages/MainPage/Map/MainPageMapTable.vue";
import MainPageMapDatePicker from "../../components/pages/MainPage/Map/MainPageMapDatePicker.vue";
import GraphSource from "../../components/graphs/GraphSource.vue";
</script>
