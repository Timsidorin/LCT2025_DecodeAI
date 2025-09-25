<template>
    <q-card class="my-card shadow-6">
        <q-card-section class="row items-center q-pa-xs">
            <div class="absolute-center">
                <q-icon
                    color="grey-7"
                    size="24px"
                    class="rotate-90"
                    name="drag_indicator"
                />
            </div>
            <DropDown class="q-ml-auto"/>
        </q-card-section>
        <q-card-section>
            <GraphForCard/>
        </q-card-section>
    </q-card>
</template>

<script setup>
import DropDown from "./BaseCard/DropDown.vue";
import GraphForCard from "./BaseCard/GraphForCard.vue";

const props = defineProps(['title']);
</script>

<style scoped>
.my-card {
    border-radius: 15px;
    max-height: 300px;
}
</style>
