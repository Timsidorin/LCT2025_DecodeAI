import {createApp} from "vue";
import App from "./App.vue";
import {router} from "./router/index.js";
import {Quasar} from 'quasar';
import "quasar/src/css/index.sass";
import "@quasar/extras/material-icons/material-icons.css";
import "../css/app.css";
import langRu from "quasar/lang/ru";

const app = createApp(App);
app.use(router)
    .use(Quasar, {
        lang: langRu
    })
    .mount('#app');
