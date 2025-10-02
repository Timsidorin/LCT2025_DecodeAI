import {defineStore} from "pinia";
import {ref} from "vue";

export const useTemplateStore = defineStore('template', () => {
    const template = ref(null);

    function setTemplate(newTemplate) {
        template.value = newTemplate;
    }

    return {template, setTemplate};
});
