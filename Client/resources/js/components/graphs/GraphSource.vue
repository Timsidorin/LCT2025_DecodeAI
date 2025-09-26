<template>
    <base-graph :column="column" type="category" :series="series" :legend="legend"/>
</template>

<script setup>
import BaseGraph from "./BaseGraph.vue";

const column = ['Положительно', 'Нейтрально', 'Отрицательно']
const series = [
    {
        data: [120, 200, 150],
        type: 'bar',
        name: 'Банки.ру'
    },
    {
        data: [120, 200, 150],
        type: 'bar',
        name: 'Сравни.ру'
    },
]
const legend = {
    data: ['Банки.ру', 'Сравни.ру'],
}
</script>
