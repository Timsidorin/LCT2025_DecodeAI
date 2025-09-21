<template>
    <div class="q-ml-xl">
        <router-link
            v-for="router in routerTabs"
            :to="router.href"
            :key="router.href"
            class="nav-link"
            active-class="nav-active-link"
        >
            <q-btn flat :label="router.label" :key="router.href"/>
        </router-link>
    </div>
</template>

<script setup>
let routerTabs = [
    {
        label: "Карта регионов",
        href: '/main/map',
    },
    {
        label: "Обзор",
        href: '/main/review',
    },
    {
        label: "Дашборд",
        href: '/main/dashboard',
    }
];
</script>

<style scoped>
.nav-link {
    color: #4e4a4a;
}

.nav-active-link {
    color: var(--gazprom-color);
}
</style>
