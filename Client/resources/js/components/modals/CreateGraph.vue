<template>
    <q-dialog v-model="model">
        <q-card style="width: 500px">
            <q-card-section>
                <h6 class="q-ma-none">Создать новый график</h6>
            </q-card-section>
            <q-card-actions class="column">
                <q-input v-model="name" style="min-width: 300px" filled label="Название"/>
                <q-select
                    class="q-mt-md"
                    style="min-width: 300px"
                    filled
                    v-model="data"
                    :options="dataList"
                    label="Данные"
                    lazy-rules
                />
<!--                На данном этапе пока не надо-->
<!--                <q-select-->
<!--                    class="q-mt-md"-->
<!--                    style="min-width: 300px"-->
<!--                    filled-->
<!--                    v-model="selectedTonality"-->
<!--                    :options="tonality"-->
<!--                    label="Тональность"-->
<!--                    lazy-rules-->
<!--                />-->
                <q-select class="q-mt-md" style="min-width: 300px" filled v-model="component" :options="componentList" label="Тип графика"/>
            </q-card-actions>
            <q-card-actions class="row justify-center">
                <div>
                    <q-btn @click="createElement" class="q-mt-md" color="primary" style="min-width: 150px">
                        Создать
                        <q-tooltip>
                            <span class="text-body2">Данный график будет отображать динамику по 3 тональностям</span>
                        </q-tooltip>
                    </q-btn>
                </div>
            </q-card-actions>
        </q-card>
    </q-dialog>
</template>

<script setup>
import {ConstructorApi} from "../../providers/ConstructorApi.js";
import {useTemplateStore} from "../../store/SelectedTemplate.js";
import {onMounted, ref} from "vue";
import {ProductApi} from "../../providers/ProductApi.js";
import {getFirstCharInUp} from "../../utils/mix.js";

const store = useTemplateStore();
const model = defineModel();
const api = new ConstructorApi();
const tonality = [
    {
        label: 'Позитивная',
        value: 'positive'
    },
    {
        label: 'Нейтральная',
        value: 'neutral'
    },
    {
        label: 'Отрицательная',
        value: 'negastive'
    }
]
const selectedTonality = ref(null);

const dataList = ref([]);
const data = ref(null);

const component = ref(null);
const name = ref('');

const componentList = [
    {
        label: 'Линейный',
        value: 'line'
    }
];
async function createElement() {
    try {
        await api.createElement({
            id: store.template.value, json: {
                type: 'graph', data: data.value, component: component.value,
                name: name.value
            }
        });
        model.value = false;
        name.value = '';
        emit('created');
    } catch (e) {
        return e;
    }
}

async function getProductList() {
    try {
        const api = new ProductApi();
        let response = await api.getListProduct();
        dataList.value = response.data.products_analysis.map(element => {
            return {
                label: getFirstCharInUp(element.product),
                value: getFirstCharInUp(element.product),
            }
        });

    } catch (e) {
        return e;
    }
}

onMounted(async () => {
   await getProductList();
});
</script>

<style scoped>

</style>
