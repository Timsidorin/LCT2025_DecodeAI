<template>
    <div class="q-pa-md">
        <q-btn-dropdown text-color="black" color="white" label="Источники данных">
            <q-list>
                <q-item v-for="element in source" clickable v-close-popup>
                    <q-item-section>
                        <q-item-label>
                            <a class="text-black" target="_blank" :href="element.url">{{element.name}}</a>
                        </q-item-label>
                    </q-item-section>
                </q-item>
            </q-list>
        </q-btn-dropdown>
    </div>
</template>

<script setup>

const source = [
    {
        name: 'Банки.ру',
        url: 'https://www.banki.ru/services/responses/bank/gazprombank/'
    },
    {
        name: 'Сравни.ру',
        url: 'https://www.sravni.ru/bank/gazprombank/'
    },
]

</script>

<style scoped>

</style>
