import LoginPage from "../pages/LoginPage.vue";
import MainPage from "../pages/MainPage.vue";
import ErrorPage from "../pages/ErrorPage.vue";
import MainPageMap from "../pages/MainPage/MainPageMap.vue";
import MainPageReview from "../pages/MainPage/MainPageReview.vue";
import MainPageDashboard from "../pages/MainPage/MainPageDashboard.vue";
import GroupButtonsMap from "../components/pages/MainPage/Header/GroupButtonsMap.vue";
import GroupButtonsReview from "../components/pages/MainPage/Header/GroupButtonsReview.vue";
import GroupButtonsDashboard from "../components/pages/MainPage/Header/GroupButtonsDashboard.vue";

export const routes = [
    {
        path: '/login',
        component: LoginPage,
    },
    {
        path: '/main',
        component: MainPage,
        name: 'main',
        redirect: '/main/map',
        children: [
            {
                path: 'map',
                components: {
                    default: MainPageMap,
                    groupButtons: GroupButtonsMap
                }
            },
            {
                path: 'review',
                components: {
                    default: MainPageReview,
                    groupButtons: GroupButtonsReview
                }
            },
            {
                path: 'dashboard',
                components: {
                    default: MainPageDashboard,
                    groupButtons: GroupButtonsDashboard
                }
            }
        ]
    },
    {
        path: '/error',
        component: ErrorPage,
        name: 'error'
    }
]
