<template>
    <div class="column q-gutter-xl">
        <BasicSummary/>
        <div class="row q-gutter-x-xl">
            <GraphDynamicsOfChanges class="col"/>
            <GraphTonalityForAllProducts class="col"/>
        </div>
        <div class="row q-gutter-x-xl">
            <MainPageTable class="col"/>
        </div>
    </div>
</template>

<script setup>
import GraphDynamicsOfChanges from "../../components/graphs/GraphDynamicsOfChanges.vue";
import GraphTonalityForAllProducts from "../../components/graphs/GraphTonalityForAllProducts.vue";
import MainPageTable from "../../components/pages/MainPage/MainPageTable.vue";
import BasicSummary from "../../components/graphs/BasicSummary.vue";
</script>

<style scoped>

</style>
