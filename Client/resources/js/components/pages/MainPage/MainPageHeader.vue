<template>
    <q-layout view="hHh lpR fFf">
        <q-header style="background-color: white" elevated class="text-white" height-hint="98">
            <q-toolbar style="background-color: var(--gazprom-color)">
                <q-toolbar-title class="q-ml-xl">
                    Газпромбанк.тех
                </q-toolbar-title>
                <MainPageHeaderUrlsSources/>
            </q-toolbar>
            <RouterTabs/>
            <div id="lower-toolbar">
                <router-view name="groupButtons"></router-view>
            </div>
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
import MainPageHeaderUrlsSources from "./Header/MainPageHeaderUrlsSources.vue";
</script>

<style>
hr {
    margin-top: 0;
    border: none;
    border-top: 1px solid #bcbcbc;
}
</style>
