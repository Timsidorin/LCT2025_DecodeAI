<template>
    <div class="q-pa-md absolute-center" style="width: 400px">
        <q-form
            @submit.prevent="onSubmit"
            class="q-gutter-md"
        >
            <q-input
                filled
                v-model="login"
                label="Логин *"
                lazy-rules
                :rules="[ val => val && val.length > 0 || 'Введите логин']"
            />
            <q-input
                filled
                type="password"
                v-model="password"
                label="Пароль *"
                lazy-rules
                :rules="[ val => val && val.length > 0 || 'Введите пароль']"
            />
            <div class="row justify-center">
                <q-btn
                    label="Войти"
                    type="submit"
                    color="primary"
                    class="width-70"
                />
            </div>
        </q-form>
    </div>
</template>

<script setup>
import {ref} from 'vue';
import {AuthApi} from "../../../providers/AuthApi.js";
import {useRouter} from "vue-router";

let login = ref('');
let password = ref('');
let router = useRouter();

async function onSubmit() {
    let auth = new AuthApi()
    auth.login({login: login.value, password: password.value})
        .then(() => {
            router.push('/main')
        })
        .catch(e => {console.log(e)});
}

</script>

<style scoped>
.width-70 {
    width: 70%;
}
</style>
