<template>
    <q-btn color="primary" label="Создать">
        <q-menu>
            <q-list style="min-width: 100px">
                <q-item clickable>
                    <q-item-section @click="modelTitle = !modelTitle">Заголовок</q-item-section>
                </q-item>
                <hr class="q-mb-none"/>
                <q-item clickable>
                    <q-item-section @click="modelGraph = !modelGraph">График</q-item-section>
                </q-item>
            </q-list>
        </q-menu>
    </q-btn>
    <CreateTitle v-model="modelTitle"/>
    <CreateGraph v-model="modelGraph"/>
</template>

<script setup>
import CreateTitle from "../../modals/CreateTitle.vue";
import CreateGraph from "../../modals/CreateGraph.vue";
import {ref} from "vue";

const modelTitle = ref(false);
const modelGraph = ref(false);
</script>

<style scoped>

</style>
