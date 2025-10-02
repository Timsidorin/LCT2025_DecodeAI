<template>
    <q-card class="not-found-card">
        <div class="column items-center q-pa-md relative-position">
            <div class="decorative-circle circle-1"></div>
            <div class="decorative-circle circle-2"></div>
            <div class="decorative-circle circle-3"></div>
            <div class="row justify-center">
                <h2 class="text-h5 text-weight-bold text-grey-9 q-mb-md">
                    Данные не найдены
                </h2>
            </div>
        </div>
    </q-card>
</template>

<script setup>
</script>

<style scoped>

.not-found-card {
    border-radius: 16px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}

.decorative-circle {
    position: absolute;
    border-radius: 50%;
    opacity: 0.1;
    pointer-events: none;
}

.circle-1 {
    width: 120px;
    height: 120px;
    background: #25689f;
    top: -40px;
    right: -40px;
}

.circle-2 {
    width: 80px;
    height: 80px;
    background: #25689f;
    bottom: 20px;
    left: -20px;
}

.circle-3 {
    width: 60px;
    height: 60px;
    background: #25689f;
    bottom: -20px;
    right: 30%;
}
</style>
