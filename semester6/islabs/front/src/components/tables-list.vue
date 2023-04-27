<template>
  <n-card class="max-w-xs">
    <ul class="flex flex-col gap-4">
      <li v-for="table in tables" :key="table">
        <n-button
          class="w-full"
          size="large"
          @click="$emit('click', table)"
          :type="activeTable === table ? 'success' : 'default'"
        >
          {{ table }}
        </n-button>
      </li>
    </ul>
  </n-card>
</template>

<script setup lang="ts">
  import { defineProps, defineEmits } from 'vue';
  import { NCard, NButton } from 'naive-ui';

  defineProps<{
    tables: string[];
    activeTable?: string;
  }>();

  defineEmits<(event: 'click', table: string) => void>();
</script>
