<script setup lang="ts">
  import { connectDb } from '@/core/use-cases/db-connection'
  import { fetchDatabases } from '@/core/use-cases/fetch-databases'
  import { NButton, NList, NListItem, NSpace } from 'naive-ui'
  import { ref } from 'vue'
  import { checkConnection } from '@/core/use-cases/get-connection-status'

  const databasesList = ref()

  console.log(checkConnection())

  if (!checkConnection()) {
    connectDb({
      host: '10.1.0.5',
      port: '5432',
      user: 'admin',
      password: 'a1128f6',
      database: 'admin',
    }).catch(reason => console.log(reason))

    console.log(checkConnection())
  }

  const connect = () => {
    fetchDatabases()
      .then(data => {
        databasesList.value = data
        console.log(data)
      })
      .catch(reason => console.log(reason))
  }
</script>
<template>
  <NSpace>
    <div class="flex flex-col">
      <NButton v-on:click="connect" type="tertiary">Получить список БД</NButton>

      <NList class="mt-10">
        <template #header>Список баз данных</template>
        <NListItem v-for="it in databasesList">
          {{ it }}
        </NListItem>
      </NList>
    </div>
  </NSpace>
</template>
