<template>
    <q-dialog v-model="model">
        <q-card style="width: 500px">
            <q-card-section>
                <h6 class="q-ma-none">Создать новый график</h6>
            </q-card-section>
            <q-card-actions class="column">
                <q-input v-model="name" style="min-width: 300px" filled label="Название"/>
                <q-select class="q-mt-md" style="min-width: 300px" filled v-model="product" :options="productList" label="Продукт"/>
                <q-select class="q-mt-md" style="min-width: 300px" filled v-model="component" :options="componentList" label="Тип графика"/>
            </q-card-actions>
            <q-card-actions class="row justify-center">
                <q-btn @click="createElement" class="q-mt-md" color="primary" style="min-width: 150px">Создать</q-btn>
            </q-card-actions>
        </q-card>
    </q-dialog>
</template>

<script setup>
import {ConstructorApi} from "../../providers/ConstructorApi.js";
import {useTemplateStore} from "../../store/SelectedTemplate.js";
import {onMounted, ref} from "vue";
import {useProductStore} from "../../store/SelectProduct.js";
import {ProductApi} from "../../providers/ProductApi.js";
import {getFirstCharInUp} from "../../utils/mix.js";

const productStore = useProductStore();
const store = useTemplateStore();
const model = defineModel();
const api = new ConstructorApi();


const productList = ref([]);
const product = ref(null);

const component = ref(null);
const name = ref('');

const componentList = [
    {
        label: 'Линейный',
        value: 'line'
    }
];
async function createElement() {
    console.log(productStore.product);
    // try {
    //     await api.createElement({
    //         id: store.template.value, json: {
    //             type: 'graph', product: product, component: component,
    //             name: name
    //         }
    //     });
    //     model.value = false;
    //     name.value = '';
    //     emit('created');
    // } catch (e) {
    //     return e;
    // }
}

async function getProductList() {
    try {
        const api = new ProductApi();
        let response = await api.getListProduct();
        productList.value = response.data.products_analysis.map(element => {
            return {
                label: getFirstCharInUp(element.product),
                value: getFirstCharInUp(element.product),
            }
        });

    } catch (e) {
        return e;
    }
}

onMounted(async () => {
   await getProductList();
});
</script>

<style scoped>

</style>
