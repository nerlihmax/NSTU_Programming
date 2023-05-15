<template>
  <n-card class="max-w-xs">
    <ul class="flex flex-col gap-4">
      <li v-for="entity in entities" :key="entity">
        <n-button
          class="w-full"
          size="large"
          @click="$emit('click', entity)"
          :type="activeEntity === entity ? 'success' : 'default'"
        >
          {{ entity }}
        </n-button>
      </li>
    </ul>
  </n-card>
</template>

<script setup lang="ts">
  import { defineEmits, defineProps } from 'vue';
  import { NButton, NCard } from 'naive-ui';

  defineProps<{
    entities: string[];
    activeEntity?: string;
  }>();

  defineEmits<(event: 'click', table: string) => void>();
</script>
