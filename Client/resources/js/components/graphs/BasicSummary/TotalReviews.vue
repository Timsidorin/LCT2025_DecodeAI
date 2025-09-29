<template>
    <q-card style="border-radius: 10px">
        <q-card-section>
            <div class="text-h6">Сводка по отзывам</div>
        </q-card-section>
        <q-card-section>
            <div>
                <div ref="chartDom" style="width: 600px; height: 500px;"></div>
            </div>
        </q-card-section>
    </q-card>
</template>

<script>
import * as echarts from 'echarts';

export default {
    name: 'PieChart',
    props: {
        chartData: {
            type: Array,
            default: () => [
                {value: 1048, name: 'Положительные', itemStyle: {color: '#2ea81d'}},
                {value: 735, name: 'Нейтральные', itemStyle: {color: '#cc0909'}},
                {value: 580, name: 'Отрицательные', itemStyle: {color: '#ffde00'}},
            ]
        },
    },
    data() {
        return {
            myChart: null
        };
    },
    mounted() {
        this.initChart();
    },
    methods: {
        initChart() {
            this.myChart = echarts.init(this.$refs.chartDom);

            const option = {
                tooltip: {
                    trigger: 'item'
                },
                legend: {},
                series: [
                    {
                        type: 'pie',
                        radius: '50%',
                        data: this.chartData,
                    }
                ]
            };

            this.myChart.setOption(option);

            // Обработчик изменения размера окна
            window.addEventListener('resize', this.handleResize);
        },
        handleResize() {
            if (this.myChart) {
                this.myChart.resize();
            }
        }
    },
    watch: {
        chartData: {
            handler(newData) {
                if (this.myChart) {
                    this.myChart.setOption({
                        series: [{
                            data: newData
                        }]
                    });
                }
            },
            deep: true
        },
        title(newTitle) {
            if (this.myChart) {
                this.myChart.setOption({
                    title: {
                        text: newTitle
                    }
                });
            }
        },
        subtext(newSubtext) {
            if (this.myChart) {
                this.myChart.setOption({
                    title: {
                        subtext: newSubtext
                    }
                });
            }
        }
    }
};
</script>
