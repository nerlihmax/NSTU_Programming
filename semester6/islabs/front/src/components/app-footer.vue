<template>
  <footer
    class="px-4 py-2 md:px-8 md:py-4 flex flex-row justify-between items-center"
  >
    <span>
      сделано с ♥️ командой
      <a
        class="text-blue-700"
        href="https://github.com/kheynov/NSTU_Programming/tree/master/semester6/islabs"
      >
        шальные ананасики
      </a>
    </span>
    <n-switch
      :value="theme === 'dark'"
      @update:value="toggled => (theme = toggled ? 'dark' : 'light')"
      size="large"
    >
      <template #icon>
        <n-icon :component="DarkTheme24Regular" />
      </template>
    </n-switch>
  </footer>
</template>

<script setup lang="ts">
  import { NSwitch, NIcon } from 'naive-ui';
  import { DarkTheme24Regular } from '@vicons/fluent';
  import { theme } from '@/stores/theme';
</script>
