<template>
    <div ref="body">
        <div v-for="item in items" :key="item.id" class="dynamic-child">
            <component :is="item.type">{{ item.title }}</component>
        </div>
    </div>
</template>

<script setup>
import {useTemplateStore} from "../../../store/SelectedTemplate.js";
import {computed, ref, watch} from 'vue';

const store = useTemplateStore();

const tempate = computed(() => {
   return store.template;
});

const body = ref('body');
const items = ref([]);

function parse(raw) {
    items.value = [];
    raw.elements.forEach((element) => {
       let domElement = JSON.parse(element.data);
        if (domElement[0] && domElement[0].type === 'h4' ) {
            parseTitle(domElement[0])
        }
    });
}

function parseTitle(title) {
    items.value.push({type: 'h4', title: title.text});
}

watch(tempate, (n, o) => {
    if (n) {
        parse(n)
    }
});
</script>

<style scoped>

</style>
