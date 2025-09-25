<template>
    <div class="row">
        <div class="q-pa-md" style="max-width: 300px">
            <q-input filled v-model="startDate" mask="date">
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
        <!--До какого числа фильтровать-->
        <div class="q-pa-md" style="max-width: 300px">
            <q-input filled v-model="endDate" mask="date">
                <template v-slot:append>
                    <q-icon name="event" class="cursor-pointer">
                        <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                            <q-date v-model="endDate">
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
let startDate = '2019-02-01 12:44';
let endDate = '2019-02-01 12:44';
</script>

<style scoped>

</style>
