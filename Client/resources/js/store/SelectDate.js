import {defineStore} from "pinia";
import {ref} from "vue";

export const useSelectDateStore = defineStore('date', () => {
    const startDate = ref('2024-01-01');
    const endDate = ref(new Date().toISOString().split('T')[0]);

    function setStartDate(newDate) {
        startDate.value = newDate;
    }

    function setEndDate(newDate) {
        endDate.value = newDate;
    }

    return {startDate, endDate, setStartDate, setEndDate};
});
