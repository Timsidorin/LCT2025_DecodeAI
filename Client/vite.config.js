import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import laravel from "laravel-vite-plugin";
import {quasar} from "@quasar/vite-plugin";

export default defineConfig({
    plugins: [
        vue(),
        laravel([
            'resources/css/app.css',
            'resources/js/app.js',
        ]),
        quasar()
    ],
    define: {
        __BASE__LARAVEL__URL__: JSON.stringify('https://lct-2025'),
        __BASE__PYTHON__URL__: JSON.stringify('https://pulseai.knastu.ru/BI'),
    },
})
