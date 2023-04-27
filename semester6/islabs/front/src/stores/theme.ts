import { effect, ref } from 'vue';

type Theme = 'light' | 'dark';

const query = window.matchMedia('(prefers-color-scheme: dark)');

const stored = window.localStorage.getItem('theme') as Theme | null;

const initial = stored ?? (query.matches ? 'dark' : 'light');

export const theme = ref<Theme>(initial);

effect(() => {
  if (theme.value === 'dark') {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
});

effect(() => {
  window.localStorage.setItem('theme', theme.value);
});
