<template>
<q-dialog v-model="model">
    <q-card style="width: 500px">
        <q-card-section>
            <h6 class="q-ma-none">Создать новый шаблон</h6>
        </q-card-section>
        <q-card-section>
            <q-input filled label="Название" v-model="name"></q-input>
        </q-card-section>
        <q-card-actions class="row justify-center">
            <q-btn
                @click="createTemplate"
                :disable="!name"
                style="width: 70%"
                class="q-ma-md"
                color="primary">
                Создать
            </q-btn>
        </q-card-actions>
    </q-card>
</q-dialog>
</template>

<script setup>
import {ref} from "vue";
import {ConstructorApi} from "../../providers/ConstructorApi.js";

let api = new ConstructorApi();
const model = defineModel();
const name = ref('');
const emit = defineEmits(['created'])

async function createTemplate() {
    try {
        await api.createTemplate({name: name.value});
        model.value = false;
        name.value = '';
        emit('created')
    } catch (e) {
        return e;
    }
}
</script>

<style scoped>

</style>
