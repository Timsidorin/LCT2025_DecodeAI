<template>
    <q-table
        :rows="rows"
        :columns="columns"
        row-key="id"
        virtual-scroll
        style="height: 300px"
    >
        <template v-slot:body="props">
            <q-tr :props="props">
                <q-td key="name" :props="props">
                    {{ props.row.name }}
                </q-td>
                <q-td key="positive" :props="props">
                    {{ props.row.positive }}
                </q-td>
                <q-td key="neutral" :props="props">
                    {{ props.row.neutral }}
                </q-td>
                <q-td key="negative" :props="props">
                    {{ props.row.negative }}
                </q-td>
            </q-tr>
        </template>
    </q-table>
</template>

<script>
import {ref} from 'vue';
const columns = [
    {name: 'name', field: 'name', align: 'center', label: 'Продукт', sortable: true,},
    {name: 'positive', field: 'positive', align: 'center', label: 'Положительно', sortable: true},
    {name: 'neutral', field: 'neutral', align: 'center', label: 'Нейтрально', sortable: true},
    {name: 'negative', field: 'negative', align: 'center', label: 'Отрицательно', sortable: true},
];

const rows = [
    {
        name: 'Кредитные карты',
        positive: 15,
        neutral: 3,
        negative: 2,
        id: 1
    },
    {
        name: 'Дебетовые карты',
        positive: 12,
        neutral: 5,
        negative: 1,
        id: 2
    },
    {
        name: 'Кредит',
        positive: 8,
        neutral: 4,
        negative: 3,
        id: 3
    },
    {
        name: 'Ипотека',
        positive: 6,
        neutral: 7,
        negative: 4,
        id: 4
    },
    {
        name: 'Кредитные карты',
        positive: 15,
        neutral: 3,
        negative: 2,
        id: 5
    },
    {
        name: 'Дебетовые карты',
        positive: 12,
        neutral: 5,
        negative: 1,
        id: 6
    },
    {
        name: 'Кредит',
        positive: 8,
        neutral: 4,
        negative: 3,
        id: 7
    }
];
const startDate = ref('01.01.2024');
const endDate = ref('23.09.2025');

export default {
    setup() {
        return {
            columns,
            rows,
            startDate,
            endDate
        }
    }
}
</script>
