<template>
  <n-card class="max-w-xs">
    <div class="flex flex-col" v-for="entity in entities" :key="entity">
      <n-button
        class="w-full my-1"
        size="large"
        @click="$emit('click', entity)"
        :type="activeEntity === entity ? 'success' : 'default'"
      >
        {{ entity }}
      </n-button>
    </div>
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
