import {
  type RouteRecordRaw,
  createRouter,
  createWebHistory,
} from 'vue-router';
import HomeView from '@/views/home-view.vue';
import TablesSelectView from '@/views/tables-select-view.vue';
import AuthView from '@/views/auth-view.vue';
import TableView from '@/views/table-view.vue';
import DatabaseLayout from '@/layouts/database-layout.vue';

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: HomeView,
  },
  {
    path: '/authorize',
    component: AuthView,
  },
  {
    path: '/database',
    component: DatabaseLayout,
    children: [
      {
        path: '',
        component: TablesSelectView,
      },
      {
        path: ':table',
        component: TableView,
      },
    ],
  },
];

export const useRouter = () =>
  createRouter({
    history: createWebHistory(),
    routes,
  });
