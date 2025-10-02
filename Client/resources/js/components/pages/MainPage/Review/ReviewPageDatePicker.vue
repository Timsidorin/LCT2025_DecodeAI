<template>
    <div class="col-auto row q-col-gutter-md q-ml-xl">
        <div class="col-auto">
            <span class="text-subtitle1">Начало периода</span>
            <q-input dense filled v-model="startDate" mask="date" style="min-width: 200px">
                <template v-slot:append>
                    <q-icon name="event" class="cursor-pointer">
                        <q-popup-proxy ref="startDateProxy" cover transition-show="scale" transition-hide="scale">
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
    </div>
    <div class="col-auto row q-col-gutter-md q-ml-xl">
        <div class="col-auto">
            <span class="text-subtitle1">Конец периода</span>
            <q-input dense filled v-model="endDate" mask="date" style="min-width: 200px">
                <template v-slot:append>
                    <q-icon name="event" class="cursor-pointer">
                        <q-popup-proxy ref="endDateProxy" cover transition-show="scale" transition-hide="scale">
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
    </div>
</template>

<script setup>
import { useSelectDateStore } from "../../../../store/SelectDate.js";
import { computed, ref } from "vue";

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

<style scoped>
span {
    color: #4e4a4a;
}
</style>
