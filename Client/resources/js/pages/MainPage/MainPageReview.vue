<template>
    <div class="column q-gutter-xl">
        <div class="row q-gutter-x-xl">
            <GraphDynamicsOfChanges class="col-4"/>
            <GraphTonalityForAllProducts class="col-4"/>
            <div class="col-3" >
                <MainPageTable />
            </div>
        </div>
        <div v-if="listProduct.length > 0" class="row q-gutter-x-xl">
            <GraphComparison title="Сравнение отзывов" :list-product="listProduct"/>
        </div>
    </div>
</template>

<script setup>
import GraphDynamicsOfChanges from "../../components/graphs/GraphDynamicsOfChanges.vue";
import GraphTonalityForAllProducts from "../../components/graphs/GraphTonalityForAllProducts.vue";
import GraphComparison from "../../components/graphs/GraphComparison.vue";
import {ProductApi} from "../../providers/ProductApi.js";
import {onMounted, ref} from "vue";
import MainPageTable from "../../components/pages/MainPage/MainPageTable.vue";

const api = new ProductApi();
const listProduct = ref([]);
async function getListProduct() {
    try {
        let response = await api.getListProduct();
        listProduct.value = response.data.products_analysis.map((element) => {
            return {
                label: element.product,
                value: element.product
            }
        });
    } catch (e) {
        return e;
    }
}

onMounted(async () => {
   await getListProduct();
});
</script>

<style scoped>

</style>
