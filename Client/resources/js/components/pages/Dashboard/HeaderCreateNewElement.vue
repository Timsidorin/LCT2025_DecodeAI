<template>
    <q-btn color="primary" label="Создать">
        <q-menu>
            <q-list style="min-width: 100px">
                <q-item clickable>
                    <q-item-section @click="modelTitle = !modelTitle">Заголовок</q-item-section>
                </q-item>
                <hr class="q-mb-none"/>
                <q-item clickable>
                    <q-item-section>График</q-item-section>
                </q-item>
            </q-list>
        </q-menu>
    </q-btn>
    <CreateTitle v-model="modelTitle"/>
</template>

<script setup>
import CreateTitle from "../../modals/CreateTitle.vue";
import {ref} from "vue";

const modelTitle = ref(false);
</script>

<style scoped>

</style>
