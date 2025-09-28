<template>
    <q-card class="custom-card">
        <q-card-section>
            <div class="text-h6">{{title}}</div>
        </q-card-section>
        <q-card-section class="scroll-content">
            <div class="q-pa-md">
                <q-option-group
                    :options="listProduct"
                    type="checkbox"
                    v-model="selectedProduct"
                />
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
import {computed, ref} from "vue";

const props = defineProps(['title', 'type', 'list-product']);
const selectedProduct = ref([]);
const statusModel = ref(false);

const disabledStatus = computed(() => {
    return selectedProduct.value.length === 0;
});
</script>

<style scoped>
.custom-card {
    border-radius: 10px;
    max-height: 400px;
    display: flex;
    flex-direction: column;
}

.scroll-content {
    overflow: auto;
    flex-grow: 1;
}
</style>
