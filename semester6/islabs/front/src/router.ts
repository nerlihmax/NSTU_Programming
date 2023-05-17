import {
  createRouter,
  createWebHistory,
  type RouteRecordRaw,
} from 'vue-router';
import HomeView from '@/views/home-view.vue';
import TablesSelectView from '@/views/tables-select-view.vue';
import AuthView from '@/views/auth-view.vue';
import TableView from '@/views/table-view.vue';
import DatabaseLayout from '@/layouts/database-layout.vue';
import DatabasesSelectView from '@/views/databases-select-view.vue';
import BackupSelectView from '@/views/backup-select-view.vue';

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
    path: '/databases',
    component: DatabaseLayout,
    children: [
      {
        path: '',
        component: DatabasesSelectView,
      },
      {
        path: ':database',
        children: [
          {
            path: 'tables',
            children: [
              {
                path: '',
                component: TablesSelectView,
              },
              {
                path: ':table',
                children: [
                  {
                    path: '',
                    component: TableView,
                  },
                  {
                    path: 'backups',
                    component: BackupSelectView,
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
];

export const useRouter = () =>
  createRouter({
    history: createWebHistory(),
    routes,
  });
