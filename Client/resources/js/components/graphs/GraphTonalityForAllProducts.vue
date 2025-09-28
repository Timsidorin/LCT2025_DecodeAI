<template>
    <base-graph title-card="Тональность по всем продуктам" :matrix="matrix" :series="series"></base-graph>
</template>

<script setup>
//TODO: ХЗ ПОД ВОПРОСОМ ЭТОТ ГРАФИК
import BaseGraph from "./BaseGraph.vue";
const matrix = {
    x: {
        data: [
            {
                value: 'Primary School',
                children: Array.from({ length: 5 }, (_, i) => {
                    return `Grade ${i + 1}`;
                })
            },
            {
                value: 'High School',
                children: Array.from({ length: 4 }, (_, i) => {
                    return `Grade ${i + 6}`;
                })
            }
        ]
    },
    y: {
        data: Array.from({ length: 6 }, (_, i) => {
            return `Class ${i + 1}`;
        })
    },
};
const series = [];
const xCnt = 9;
const yCnt = 6;
for (let i = 0; i < xCnt; ++i) {
    for (let j = 0; j < yCnt; ++j) {
        series.push({
            type: 'pie',
            coordinateSystem: 'matrix',
            center: [`Grade ${i + 1}`, `Class ${j + 1}`],
            radius: 18,
            data: [
                {
                    value: Math.round(Math.random() * 10) + 10,
                    name: 'Male'
                },
                {
                    value: Math.round(Math.random() * 10) + 10,
                    name: 'Female'
                }
            ],
            label: {
                show: false
            },
            emphasis: {
                label: {
                    show: false
                }
            }
        });
    }
}
</script>

<style scoped>

</style>
