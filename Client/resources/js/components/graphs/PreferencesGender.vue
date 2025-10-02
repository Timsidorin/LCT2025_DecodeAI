<template>
    <div style="height: 200px" class="row q-gutter-x-xl">
        <GenderData v-if="maleData" gender="male" :insights="insights" :data="maleData"/>
        <GenderData v-if="femaleData" gender="female" :insights="insights" :data="femaleData"/>
    </div>
</template>

<script setup>
import {StatisticApi} from "../../providers/StatisticApi.js";
import {onMounted, ref} from "vue";
import GenderData from "./PreferencesGender/GenderData.vue";

const maleData = ref(null);
const femaleData = ref(null);
const api = new StatisticApi();
const insights = ref(null)

async function getData() {
    try {
        let response = await api.getGenderProductPrefereces();
        maleData.value = response.data.male_preferences;
        femaleData.value = response.data.female_preferences;
        insights.value = response.data.insights;
    } catch (e) {
        return e;
    }
}

onMounted(async () => {
   await getData();
});


</script>

<style scoped>

</style>
