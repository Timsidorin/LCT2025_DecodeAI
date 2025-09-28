import {defineStore} from "pinia";
import {ref} from "vue";

export const useRegionStore = defineStore('region', () => {
    const region = ref(JSON.parse(localStorage.getItem('selectedRegion')) ?? null);

    function setRegion(newRegion) {
        region.value = newRegion;
        localStorage.setItem('selectedRegion', JSON.stringify(newRegion));
    }

    return {region, setRegion};
});
