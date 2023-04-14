<script setup lang="ts">
  import { connectDb } from '@/core/use-cases/db-connection'
  import { fetchDatabases } from '@/core/use-cases/fetch-databases'
  import { ref } from 'vue'
  import { checkConnection } from '@/core/use-cases/get-connection-status'
  import SideList from '@/SideList.vue'
  import { NButton } from 'naive-ui'

  const databasesList = ref()

  const authenticated = ref(checkConnection())

  const connect = async () => {
    try {
      await connectDb({
        host: 'localhost',
        port: '5432',
        user: 'postgres',
        password: 'qwertyqwerty',
        database: 'postgres',
      })
      authenticated.value = true
    } catch (error) {
      console.error(error)
      authenticated.value = false
      return
    }

    try {
      const databases = await fetchDatabases()
      databasesList.value = databases
      console.log(databases)
    } catch (error) {
      console.error(error)
    }
  }
</script>

<template>
  <div class="flex flex-row">
    <SideList
      v-if="authenticated"
      :databases-list="databasesList"
      v-on:connect="console.log('aboba')"
    />
    <div v-else>
      <NButton :onclick="connect" tertiary>Подключиться к БД</NButton>
    </div>
  </div>
</template>
