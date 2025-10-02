<template>
    <hr/>
    <div class="row items-center q-ml-xl q-pa-md">
        <div class="col-auto row q-col-gutter-md">
            <div class="col-auto row items-center">
                <div>
                    <span class="text-subtitle1">Шаблон</span>
                    <q-select
                        dense
                        filled
                        outlined
                        v-model="selectedTemplates"
                        :options="tempatesWithButton"
                        style="min-width: 200px"
                        @update:model-value="onTemplateSelect"
                    >
                        <template v-slot:option="scope">
                            <q-item v-bind="scope.itemProps">
                                <q-item-section>
                                    <q-item-label v-if="scope.opt.isButton">
                                        <q-btn
                                            flat
                                            dense
                                            @click="statusModalCreate = !statusModalCreate"
                                            style="width: 100%; justify-content: center;"
                                        >
                                            + Добавить шаблон
                                        </q-btn>
                                    </q-item-label>
                                    <q-item-label v-else>
                                        {{ scope.opt.label }}
                                    </q-item-label>
                                </q-item-section>
                            </q-item>
                        </template>
                    </q-select>
                </div>
                <q-icon @click="deleteTemplate(selectedTemplates.value)" class="q-mt-lg q-ml-md button-trash" size="26px" name="delete">
                    <q-tooltip class="text-body2">
                        Удалить шаблон
                    </q-tooltip>
                </q-icon>
                <div class="column q-ml-xl">
                    <span class="text-subtitle1">Новый элемент</span>
                    <HeaderCreateNewElement/>
                </div>
            </div>
        </div>
    </div>
    <create-new-template @created="getTemplates" v-model="statusModalCreate"/>
</template>

<script setup>
import {ConstructorApi} from "../../../../providers/ConstructorApi.js";
import {onMounted, ref, computed} from "vue";
import CreateNewTemplate from "../../../modals/CreateNewTemplate.vue";
import {useTemplateStore} from "../../../../store/SelectedTemplate.js";
import HeaderCreateNewElement from "../../Dashboard/HeaderCreateNewElement.vue";

const api = new ConstructorApi();
const tempates = ref([]);

const selectedTemplates = computed({
    get: () => store.template,
    set: (value) => store.setTemplate(value)
});

const statusModalCreate = ref(false);
const store = useTemplateStore();

const tempatesWithButton = computed(() => {
    const templatesList = tempates.value.map(element => ({
        ...element,
        isButton: false
    }));

    return [
        ...templatesList,
        { label: 'Добавить шаблон', value: 'add-template', isButton: true }
    ];
});

async function getTemplates() {
    try {
        let response = await api.getTemplates();
        tempates.value = response.data.map((element) => {
            return {label: element.name, value: element.id, elements: element.elements_template}
        });
        selectedTemplates.value = tempates.value[0];
    } catch (e) {
        return e;
    }
}

function onTemplateSelect(value) {
    if (value && value.isButton) {
        selectedTemplates.value = null;
    }
}

onMounted(async () => {
    await getTemplates();
});

async function deleteTemplate(id) {
    try {
        let response = await api.deleteTemplate(id);
        await getTemplates();
    } catch (e) {
        return e;
    }
}
</script>

<style scoped>
span {
    color: #4e4a4a;
}

.button-trash {
    color: black;
}

.button-trash:hover {
    color: rgb(43, 97, 236);
    cursor: pointer;
}
</style>
