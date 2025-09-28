<template>
    <div class="col-auto row q-col-gutter-md q-ml-xl">
        <div class="col-auto">
            <span class="text-subtitle1">Продукты и услуги</span>
            <q-select dense filled outlined v-model="product" :options="listProduct" style="min-width: 200px"/>
        </div>
    </div>
</template>

<script setup>
import {ProductApi} from "../../../providers/ProductApi.js";
import {computed, onMounted, ref} from 'vue';
import {useProductStore} from "../../../store/SelectProduct.js";

const api = new ProductApi();
const listProduct = ref([]);
const store = useProductStore();

async function getData() {
    try {
        let response = await api.getListProduct();
        listProduct.value = response.data.products_analysis.map((element) => {
            return {
                label: element.product,
                value: element.product
            }
        });
        product.value = listProduct.value[0];
    } catch (e) {
        return e;
    }
}

const product = computed({
    get: () => store.product,
    set: (value) => store.setProduct(value)
});

onMounted(async () => {
    await getData();
})
</script>

<style scoped>
span {
    color: #4e4a4a;
}
</style>
