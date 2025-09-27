<template>
    <hr/>
    <div class="row items-center q-ml-xl q-pa-md">
        <div class="col-auto row q-col-gutter-md">
            <div class="col-auto">
                <span class="text-subtitle1">Регион</span>
                <q-select dense outlined v-model="model" :options="options" style="min-width: 200px"/>
            </div>
            <div class="col-auto">
                <span class="text-subtitle1">Город</span>
                <q-select dense outlined v-model="model" :options="options" style="min-width: 200px"/>
            </div>
            <div class="col-auto">
                <span class="text-subtitle1">Период</span>
                <q-select dense outlined v-model="model" :options="options" style="min-width: 200px"/>
            </div>
        </div>
        <div class="col-auto row q-col-gutter-md q-ml-xl">
            <div class="col-auto">
                <span class="text-subtitle1">Линейка</span>
                <q-select dense outlined v-model="model" :options="options" style="min-width: 200px"/>
            </div>
        </div>
    </div>
</template>

<script setup>

</script>

<style scoped>
span {
    color: #4e4a4a;
}
</style>
