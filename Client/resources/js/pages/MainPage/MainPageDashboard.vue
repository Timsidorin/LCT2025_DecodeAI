<template>
    <q-btn @click="createTemplate">fsdf</q-btn>
    <q-btn @click="getTemplates">fsdf</q-btn>
    <q-btn @click="deleteTemplate">fsdf</q-btn>
    <q-btn @click="createElement">fsdf</q-btn>
</template>

<script setup>
import {ConstructorApi} from "../../providers/ConstructorApi.js";

const api = new ConstructorApi();

async function createTemplate() {
    try {
        await api.createTemplate({name: '123'});
    } catch (e) {
        return e;
    }
}

async function getTemplates() {
    try {
        await api.getTemplates();
    } catch (e) {
        return e;
    }
}

async function deleteTemplate() {
    try {
        await api.deleteTemplate(1);
    } catch (e) {
        return e;
    }
}

async function createElement() {
    try {
        await api.createElement({id: 1});
    } catch (e) {
        return e;
    }
}
</script>


<style scoped>

</style>
