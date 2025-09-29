<template>
    <q-card class="custom-card">
        <q-card-section>
            <div class="text-h6">Сравнение продуктов</div>
        </q-card-section>
        <q-card-section class="scroll-content">
            <div class="q-pa-md">
                <q-option-group
                    :options="listProduct"
                    type="checkbox"
                    v-model="selectedProduct"
                >
                    <template v-slot:label="opt">
                        <div class="row items-center">
                            <span>{{ opt.label }}</span>
                            <div v-if="selectedProduct.includes(opt.value)">
                                <div>
                                    <q-option-group
                                        v-model="group"
                                        :options="typeComp"
                                        color="primary"
                                        inline
                                        dense
                                    />
                                </div>
                            </div>
                        </div>
                    </template>
                </q-option-group>
            </div>
        </q-card-section>
        <q-card-section class="row justify-end">
            <q-btn :disable="disabledStatus" color="primary" label="Сравнить" @click="statusModel = !statusModel"/>
        </q-card-section>
    </q-card>
    <graph-comparison-modal :product-list="selectedProduct" :title="title" v-model="statusModel"/>
</template>

<script setup>
import GraphComparisonModal from "../modals/GraphComparisonModal.vue";
import {onMounted, ref, computed} from "vue";
import {ProductApi} from "../../providers/ProductApi.js";

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

const selectedProduct = ref([]);
const statusModel = ref(false);
const typeComp = [
    {
        label: 'Положительно',
        value: ''
    },
    {
        label: 'Нейтрально',
        value: ''
    },
    {
        label: 'Отрицательно',
        value: ''
    }
]

const disabledStatus = computed(() => {
    return selectedProduct.value.length === 0;
});
</script>

<style scoped>
.custom-card {
    width: 100%;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    height: 600px;
}

.scroll-content {
    overflow: auto;
    flex-grow: 1;
}
</style>
