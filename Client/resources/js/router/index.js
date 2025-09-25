import { createWebHistory, createRouter } from 'vue-router';
import {routes} from "./routes.js";
import {AuthApi} from "../providers/AuthApi.js";

export const router = createRouter({
    history: createWebHistory(),
    routes
});

// router.beforeEach(async (to, from, next) => {
//     if (to.href.includes('/main')) {
//         let tokenAuth = localStorage.getItem('bearerToken');
//         let auth = false;
//
//         if (tokenAuth) {
//             let authClient = new AuthApi();
//             auth = await authClient.checkAuth(tokenAuth);
//         }
//         if (!auth.data?.uuid) {
//             next({ name: 'error' });
//         } else {
//             next();
//         }
//
//     } else {
//         next();
//     }
// });
