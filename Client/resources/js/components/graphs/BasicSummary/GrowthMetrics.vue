<template>
    <div class="column no-wrap">
        <q-card style="height: 50%; border-radius: 10px">
                <q-card-section>
                    <h6 class="q-mb-none q-mt-xs">Сводка за {{data.growth_metrics.period_hours}} часа</h6>
                </q-card-section>
                <q-card-section class="row wrap q-gutter-lg q-mt-md">
                    <q-card class="col" style="border-radius: 10px">
                        <q-card-section>
                            <h6 class="q-mb-none q-mt-xs text-center text-grey-7">Новых отзывов</h6>
                            <h4 class="q-mb-none text-weight-bold q-mt-xs text-center text-grey-9 row items-center justify-center">
                                <q-icon name="draw"/>
                                {{data.growth_metrics.new_reviews}}
                            </h4>
                        </q-card-section>
                    </q-card>
                    <q-card class="col" style="border-radius: 10px">
                        <q-card-section>
                            <h6 class="q-mb-none q-mt-xs text-center text-grey-7">Всего отзывов</h6>
                            <h4 class="text-weight-bold q-mb-none q-mt-xs text-center text-grey-9 row items-center justify-center">
                                <q-icon name="visibility" class="q-mr-xs"/>
                                {{data.growth_metrics.total_reviews}}
                            </h4>
                        </q-card-section>
                    </q-card>
                    <q-card style="border-radius: 10px">
                        <q-card-section class="col">
                            <h6 class="q-mb-none q-mt-xs text-center text-grey-7">
                                Рейтинг
                            </h6>
                            <h4 class="q-mb-none text-weight-bold q-mt-xs text-center text-grey-9 row items-center">
                                <q-icon name="arrow_downward" :class="{'rotate-180': data.growth_metrics.growth_rate >= 0.5}"/>
                                {{data.growth_metrics.growth_rate * 100}}%
                            </h4>
                        </q-card-section>
                    </q-card>
                </q-card-section>
            </q-card>
        <q-card class="q-mt-md" style="height: 50%; border-radius: 10px">
            <q-card-section>
                <h6 class="q-mb-none q-mt-xs">Сводка по полу</h6>
            </q-card-section>
            <q-card-section class="row wrap q-gutter-lg">
                <q-card class="col" style="border-radius: 10px">
                    <q-card-section>
                        <div class="q-mb-none text-center">
                            <q-icon class="text-center" color="pink" size="50px" name="female"/>
                            <p class="text-grey q-my-none">Женщины</p>
                        </div>
                        <h4 class="q-mb-none text-weight-bold q-mt-xs text-center text-grey-9">{{data.gender_distribution.female}}</h4>
                    </q-card-section>
                </q-card>
                <q-card class="col" style="border-radius: 10px">
                    <q-card-section>
                        <div class="q-mb-none text-center">
                            <q-icon class="text-center" color="blue" size="50px" name="male"/>
                            <p class="text-grey q-my-none">Мужчины</p>
                        </div>
                        <h4 class="text-weight-bold q-mb-none q-mt-xs text-center text-grey-9">{{data.gender_distribution.male}}</h4>
                    </q-card-section>
                </q-card>
            </q-card-section>
        </q-card>
    </div>


</template>

<script setup>
const props = defineProps(['data']);
</script>


<style scoped>

</style>
