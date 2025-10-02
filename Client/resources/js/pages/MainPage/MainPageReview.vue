<template>
    <div class="column q-gutter-xl">
        <div class="row q-gutter-x-xl">
            <PreferencesGender/>
            <MainPageTable class="col"/>
        </div>
        <div class="row q-gutter-x-xl">
            <GraphDynamicsOfChanges class="col"/>
            <GraphTonalityForAllProducts chart-height="700px" class="col"/>
        </div>
        <BasicSummary/>
    </div>
</template>

<script setup>
import GraphDynamicsOfChanges from "../../components/graphs/GraphDynamicsOfChanges.vue";
import GraphTonalityForAllProducts from "../../components/graphs/GraphTonalityForAllProducts.vue";
import MainPageTable from "../../components/pages/MainPage/MainPageTable.vue";
import BasicSummary from "../../components/graphs/BasicSummary.vue";
import PreferencesGender from "../../components/graphs/PreferencesGender.vue";
</script>

<style scoped>

</style>
