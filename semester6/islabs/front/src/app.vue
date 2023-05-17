<template>
  <n-config-provider
    class="flex flex-col h-full w-full text-black dark:text-white bg-white dark:bg-neutral-800 duration-150"
    :theme="appearanceTheme"
  >
    <n-loading-bar-provider>
      <n-message-provider>
        <router-view class="flex-grow w-full" />
        <app-footer />
      </n-message-provider>
    </n-loading-bar-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
  import { computed } from 'vue';
  import { RouterView } from 'vue-router';
  import { NConfigProvider, lightTheme, darkTheme, NLoadingBarProvider, NMessageProvider } from 'naive-ui';
  import { theme } from '@/stores/theme';
  import AppFooter from '@/components/app-footer.vue';

  const appearanceTheme = computed(() =>
    theme.value === 'light' ? lightTheme : darkTheme,
  );
</script>
