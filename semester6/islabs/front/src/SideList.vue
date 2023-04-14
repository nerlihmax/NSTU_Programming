<template>
  <NSpace>
    <div class="flex flex-col">
      <NButton v-on:click="emit('')" type="tertiary">
        Получить список БД
      </NButton>
      <!--    Create side bar in naive ui-->

      <NList class="mt-10">
        <template #header>Список баз данных</template>
        <NListItem v-for="it in databasesList" :key="it.id">
          <div class="flex flex-row">
            <p>{{ it.name }}</p>
            <NButton v-on:click="emit('connect', it.id)" type="tertiary">
              Подключиться
            </NButton>
          </div>
        </NListItem>
      </NList>
    </div>
  </NSpace>
</template>

<script setup lang="ts">
  import { NButton, NList, NListItem, NSpace } from 'naive-ui'
  import type { DatabaseEntry } from '@/types'

  defineProps<{
    databasesList: DatabaseEntry[]
  }>()

  const emit = defineEmits<{
    (e: 'connect', id: number): void
  }>()
</script>
