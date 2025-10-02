<template>
    <q-card style="border-radius: 10px; width: 100%">
        <q-card-section>
            <div class="text-h6">Фильтрация</div>
        </q-card-section>
        <q-card-section>
            <div class="row justify-center">
                <div style="max-width: 300px">
                    <q-input dense hint="Начало периода" filled v-model="startDate" mask="date">
                        <template v-slot:append>
                            <q-icon name="event" class="cursor-pointer">
                                <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                                    <q-date
                                        v-model="startDate"
                                        :locale="russianLocale"
                                        today-btn
                                        mask="YYYY-MM-DD"
                                    />
                                </q-popup-proxy>
                            </q-icon>
                        </template>
                    </q-input>
                </div>
                <div class="q-ml-xl" style="max-width: 300px">
                    <q-input dense hint="Конец периода" filled v-model="endDate" mask="date">
                        <template v-slot:append>
                            <q-icon name="event" class="cursor-pointer">
                                <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                                    <q-date
                                        v-model="endDate"
                                        :locale="russianLocale"
                                        today-btn
                                        mask="YYYY-MM-DD"
                                    />
                                </q-popup-proxy>
                            </q-icon>
                        </template>
                    </q-input>
                </div>
                <main-page-region-list class="q-mt-md col-12" style="max-width: 300px"/>
            </div>
        </q-card-section>
    </q-card>
</template>

<script setup>
import {useSelectDateStore} from "../../../store/SelectDate.js";
import {computed} from "vue";
import MainPageRegionList from "./MainPageRegionList.vue";

const store = useSelectDateStore();

// Русская локализация для Quasar
const russianLocale = {
    days: 'Воскресенье_Понедельник_Вторник_Среда_Четверг_Пятница_Суббота'.split('_'),
    daysShort: 'Вс_Пн_Вт_Ср_Чт_Пт_Сб'.split('_'),
    months: 'Январь_Февраль_Март_Апрель_Май_Июнь_Июль_Август_Сентябрь_Октябрь_Ноябрь_Декабрь'.split('_'),
    monthsShort: 'Янв_Фев_Мар_Апр_Май_Июн_Июл_Авг_Сен_Окт_Ноя_Дек'.split('_'),
    firstDayOfWeek: 1, // Понедельник
    format24h: true,
    pluralDay: 'дни'
};

const startDate = computed({
    get: () => store.startDate,
    set: (value) => {
        const formattedDate = value.replace(/\//g, '-');
        store.setStartDate(formattedDate);
    }
});

const endDate = computed({
    get: () => store.endDate,
    set: (value) => {
        const formattedDate = value.replace(/\//g, '-');
        store.setEndDate(formattedDate);
    }
});
</script>
