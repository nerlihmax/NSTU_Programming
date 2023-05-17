import { createApp } from 'vue';
import { createPinia } from 'pinia';

import { useRouter } from '@/router';
import App from './app.vue';

import './index.css';
import '@/assets/preflight.css';

createApp(App).use(createPinia()).use(useRouter()).mount('#app');
