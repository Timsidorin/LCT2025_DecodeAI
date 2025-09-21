<template>
    <q-layout view="hHh lpR fFf">
        <q-header style="background-color: white" elevated class="text-white" height-hint="98">
            <q-toolbar style="background-color: var(--gazprom-color)">
                <q-toolbar-title class="q-ml-xl">
                    Газпромбанк.тех
                </q-toolbar-title>
            </q-toolbar>
            <RouterTabs/>
        </q-header>
        <q-page-container>
            <div class="q-ma-xl">
                <RouterView/>
            </div>
        </q-page-container>
    </q-layout>
</template>

<script setup>
import RouterTabs from "./Header/RouterTabs.vue";
</script>
