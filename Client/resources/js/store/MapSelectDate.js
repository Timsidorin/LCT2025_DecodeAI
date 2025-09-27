import {defineStore} from "pinia";
import {ref} from "vue";

export const useSelectDateStore = defineStore('date', () => {
    const startDate = ref('2024-01-01');
    const endDate = ref('2025-09-27');

    function setStartDate(newDate) {
        startDate.value = newDate;
        console.log(startDate.value)
    }

    function setEndDate(newDate) {
        endDate.value = newDate;
    }

    return {startDate, endDate, setStartDate, setEndDate};
});
