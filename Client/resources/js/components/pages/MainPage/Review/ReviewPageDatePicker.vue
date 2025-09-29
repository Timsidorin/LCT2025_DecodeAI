<template>
<div class="col-auto row q-col-gutter-md q-ml-xl">
    <div class="col-auto">
        <span class="text-subtitle1">Начало периода</span>
        <q-input dense filled v-model="startDate" mask="date" style="min-width: 200px">
            <template v-slot:append>
                <q-icon name="event" class="cursor-pointer">
                    <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                        <q-date v-model="startDate">
                            <div class="row items-center justify-end">
                                <q-btn v-close-popup label="Close" color="primary" flat/>
                            </div>
                        </q-date>
                    </q-popup-proxy>
                </q-icon>
            </template>
        </q-input>
    </div>
</div>
<div class="col-auto row q-col-gutter-md q-ml-xl">
    <div class="col-auto">
        <span class="text-subtitle1">Конец периода</span>
        <q-input dense filled v-model="startDate" mask="date" style="min-width: 200px">
            <template v-slot:append>
                <q-icon name="event" class="cursor-pointer">
                    <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                        <q-date v-model="startDate">
                            <div class="row items-center justify-end">
                                <q-btn v-close-popup label="Close" color="primary" flat/>
                            </div>
                        </q-date>
                    </q-popup-proxy>
                </q-icon>
            </template>
        </q-input>
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
