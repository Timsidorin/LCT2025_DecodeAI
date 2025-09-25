<template>
    <div class="row">
        <div class="col-7">
            <main-page-map-russia />
        </div>
        <div class="q-ml-auto q-mr-xl">
            <main-page-map-date-picker/>
            <main-page-map-table/>
        </div>
    </div>
</template>

<script setup>
import MainPageMapRussia from "../../components/pages/MainPage/Map/MainPageMapRussia.vue";
import MainPageMapTable from "../../components/pages/MainPage/Map/MainPageMapTable.vue";
import MainPageMapDatePicker from "../../components/pages/MainPage/Map/MainPageMapDatePicker.vue";
</script>

<style scoped>

</style>
