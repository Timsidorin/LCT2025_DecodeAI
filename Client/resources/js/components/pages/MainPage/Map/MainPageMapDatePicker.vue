<template>
    <q-card style="border-radius: 10px">
        <q-card-section>
            <div class="text-h6">Фильтрация</div>
        </q-card-section>
        <q-card-section>
                <div class="row justify-center">
                    <div style="max-width: 300px">
                        <q-input hint="Начало периода" filled v-model="startDate" mask="date">
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
                    <div class="q-ml-xl" style="max-width: 300px">
                        <q-input hint="Конец периода" filled v-model="endDate" mask="date">
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
                    <main-page-map-region-list class="q-mt-md col-12" style="max-width: 300px"/>
                </div>
        </q-card-section>
    </q-card>
</template>

<script setup>
import {useSelectDateStore} from "../../../../store/MapSelectDate.js";
import {computed} from "vue";
import MainPageMapRegionList from "./MainPageMapRegionList.vue";

const store = useSelectDateStore();

const startDate = computed({
    get: () => store.startDate,
    set: (value) => store.setStartDate(value)
});

const endDate = computed({
    get: () => store.endDate,
    set: (value) => store.setEndDate(value)
});
</script>
