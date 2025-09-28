import {defineStore} from "pinia";
import {ref} from "vue";

export const useCityStore = defineStore('city', () => {
    const city = ref(JSON.parse(localStorage.getItem('selectedCity')) ?? null);

    function setCity(newCity) {
        city.value = newCity;
        localStorage.setItem('selectedCity', JSON.stringify(newCity));
    }

    return {city, setCity};
});
