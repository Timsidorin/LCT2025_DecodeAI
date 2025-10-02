<template>
    <div ref="body">
        <template v-for="(item, index) in items" :key="item.id">
            <!-- Текстовые элементы -->
            <component
                v-if="item.type !== 'graph'"
                :is="item.type"
                class="text-element"
            >
                {{ item.title }}
            </component>

            <!-- Группа графиков -->
            <div v-else-if="isFirstGraphInSequence(index)" class="graphs-grid">
                <LineChart
                    v-for="graph in getConsecutiveGraphs(index)"
                    :key="graph.id"
                    :title="graph.title"
                    :height="400"
                    :chart-data="graph.data"
                    :series-config="graph.seriesConfig"
                    :x-axis-config="graph.xAxisConfig"
                    :y-axis-config="graph.yAxisConfig"
                    class="chart-item"
                />
            </div>
        </template>
    </div>
</template>

<script setup>
import { useTemplateStore } from "../../../store/SelectedTemplate.js";
import { computed, ref, watch } from 'vue';
import LineChart from "../../constructor/LineChart.vue";
import { StatisticApi } from "../../../providers/StatisticApi.js";

const store = useTemplateStore();
const tempate = computed(() => {
    return store.template;
});

const body = ref('body');
const items = ref([]);

// Функция для проверки, является ли график первым в последовательности
function isFirstGraphInSequence(index) {
    if (items.value[index].type !== 'graph') return false;

    // Если это первый элемент или предыдущий элемент не график
    return index === 0 || items.value[index - 1].type !== 'graph';
}

// Функция для получения последовательных графиков
function getConsecutiveGraphs(startIndex) {
    const graphs = [];
    let i = startIndex;

    while (i < items.value.length && items.value[i].type === 'graph') {
        graphs.push(items.value[i]);
        i++;
    }

    return graphs;
}

function parse(raw) {
    items.value = [];
    raw.elements.forEach((element) => {
        let domElement = JSON.parse(element.data);
        if (domElement[0] && domElement[0].type === 'h4' ) {
            parseTitle(domElement[0])
        }
        if (domElement[0] && domElement[0].type === 'graph' ) {
            parseGraph(domElement[0])
        }
    });
}

function parseTitle(title) {
    items.value.push({
        type: 'h4',
        title: title.text,
        id: generateId()
    });
}

async function parseGraph(graph) {
    if (graph.component && graph.component.value) {
        const api = new StatisticApi();
        try {
            let response = await api.getDynamicsOfChanges(
                '2024-01-01',
                '2025-09-27',
                graph.data.value
            );

            // Преобразуем данные в нужный формат
            const transformedData = transformChartData(response.data);

            items.value.push({
                type: 'graph',
                id: generateId(),
                title: graph.name || 'График отзывов',
                data: transformedData,
                seriesConfig: [
                    {
                        name: 'Положительные отзывы',
                        dataKey: 'Positive_Reviews',
                        color: '#52c41a',
                        lineWidth: 3,
                        smooth: true,
                        showSymbol: false
                    },
                    {
                        name: 'Негативные отзывы',
                        dataKey: 'Negative_Reviews',
                        color: '#ff4d4f',
                        lineWidth: 3,
                        smooth: true,
                        showSymbol: false
                    },
                    {
                        name: 'Нейтральные отзывы',
                        dataKey: 'Neutral_Reviews',
                        color: '#faad14',
                        lineWidth: 2,
                        lineType: 'dashed',
                        smooth: true,
                        showSymbol: false
                    }
                ],
                xAxisConfig: {
                    name: 'Период',
                    dataKey: 'Month'
                },
                yAxisConfig: {
                    name: 'Количество отзывов'
                }
            });
        } catch (e) {
            console.error('Ошибка при получении данных графика:', e);
            items.value.push({
                type: 'graph',
                id: generateId(),
                title: graph.name || 'График (ошибка загрузки)',
                data: [],
                seriesConfig: [],
                xAxisConfig: { name: 'Период', dataKey: 'Month' },
                yAxisConfig: { name: 'Количество отзывов' }
            });
        }
    }
}

function transformChartData(apiData) {
    if (!apiData || apiData.length === 0) return [];

    const headers = apiData[0];
    const dataRows = apiData.slice(1);

    return dataRows.map(row => {
        const obj = {};
        headers.forEach((header, index) => {
            // Преобразуем числовые значения, если это возможно
            const value = row[index];
            obj[header] = typeof value === 'number' || !isNaN(value) ? Number(value) : value;
        });
        return obj;
    });
}

function generateId() {
    return Date.now() + Math.random().toString(36).substr(2, 9);
}

watch(tempate, (n, o) => {
    if (n) {
        parse(n)
    }
});
</script>

<style scoped>
.text-element {
    margin: 20px 0;
    width: 100%;
}

.graphs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
    gap: 20px;
    margin: 20px 0;
    width: 100%;
}

.chart-item {
    min-width: 0; /* Важно для корректного отображения в grid */
}

/* Адаптивность для разных размеров экрана */
@media (max-width: 1200px) {
    .graphs-grid {
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    }
}

@media (max-width: 768px) {
    .graphs-grid {
        grid-template-columns: 1fr;
        gap: 15px;
    }
}

@media (max-width: 480px) {
    .graphs-grid {
        gap: 10px;
    }
}
</style>
