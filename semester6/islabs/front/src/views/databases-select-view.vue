<template>
  <main class="flex flex-col items-center justify-center px-8 md:px-48">
    <entities-list :entities="databases" @click="connectToDatabase" />
  </main>
</template>

<script setup lang="ts">
  import { onMounted, ref } from 'vue';
  import { useRouter } from 'vue-router';
  import { useConnectionState } from '@/stores/connection';
  import EntitiesList from '@/components/entities-list.vue';
  import { fetchDatabases } from '@/core/use-cases/fetch-databases';

  const connection = useConnectionState();
  const router = useRouter();

  const databases = ref<string[]>([]);

  onMounted(async () => {
    databases.value = await fetchDatabases();
  });

  const connectToDatabase = async (database: string) => {
    const creds = connection.connection;
    if (!creds) return;
    await connection.disconnect();
    await connection.connect({ ...creds, database });
    await router.push(`/databases/${database}/tables`);
  };
</script>
