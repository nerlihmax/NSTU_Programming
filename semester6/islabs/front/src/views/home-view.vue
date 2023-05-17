<template>
  <main class="flex items-center justify-center flex-col gap-4">
    <img src="src/assets/logo.svg" alt="" width="250" height="250" />
    <h1 class="text-4xl">Привет, мы команда «шальные ананасики»</h1>
    <n-button size="large" @click="start">Начать работу</n-button>
  </main>
</template>

<script setup lang="ts">
  import { NButton } from 'naive-ui';
  import { useRouter } from 'vue-router';

  const router = useRouter();

  const start = () => router.push('/databases');
</script>
