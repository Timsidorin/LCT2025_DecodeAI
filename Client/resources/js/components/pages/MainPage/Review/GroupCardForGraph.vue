<template>
    <div ref="groupGraph" class="row q-gutter-xl">
        <div v-for="a in 40" :data-swapy-slot="a" class="col-2">
            <base-card-for-graph :data-swapy-item="a" :title="a"/>
        </div>
    </div>
</template>

<script setup>
import BaseCardForGraph from "./BaseCardForGraph.vue";
import {onMounted, ref} from 'vue';
import {createSwapy} from "swapy";

const groupGraph = ref(null);
const swapy = ref(null);

onMounted(() => {
    if (groupGraph.value) {
        swapy.value = createSwapy(groupGraph.value, {});
        swapy.value.onSwap(handlerSwap);
    }

    function handlerSwap(event) {
        console.log(event);
    }
});
</script>

<style scoped>

</style>
