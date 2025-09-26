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
        lang: langRu,
        config: {
            brand: {
                primary: '#2b61ec',
                secondary: '#26A69A',
                accent: '#9C27B0',

                dark: '#1d1d1d',
                'dark-page': '#121212',

                positive: '#21BA45',
                negative: '#C10015',
                info: '#31CCEC',
                warning: '#F2C037'
            }
        }
    })
    .mount('#app');
