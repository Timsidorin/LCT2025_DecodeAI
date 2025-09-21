import LoginPage from "../pages/LoginPage.vue";
import MainPage from "../pages/MainPage.vue";
import ErrorPage from "../pages/ErrorPage.vue";
import MainPageMap from "../pages/MainPage/MainPageMap.vue";
import MainPageReview from "../pages/MainPage/MainPageReview.vue";
import MainPageDashboard from "../pages/MainPage/MainPageDashboard.vue";

export const routes = [
    {
        path: '/login',
        component: LoginPage
    },
    {
        path: '/main',
        component: MainPage,
        name: 'main',
        redirect: '/main/map',
        children: [
            {
                path: 'map',
                component: MainPageMap
            },
            {
                path: 'review',
                component: MainPageReview
            },
            {
                path: 'dashboard',
                component: MainPageDashboard
            }
        ]
    },
    {
        path: '/error',
        component: ErrorPage, name: 'error'
    }
]
