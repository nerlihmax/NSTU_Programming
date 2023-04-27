import { createApp } from 'vue';
import { createPinia } from 'pinia';

import { useRouter } from '@/router';
import App from './app.vue';

import './index.css';

createApp(App).use(createPinia()).use(useRouter()).mount('#app');
