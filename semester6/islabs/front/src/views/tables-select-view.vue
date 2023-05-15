<template>
  <main class="flex flex-col items-center justify-center px-8 md:px-48">
    <entities-list :entities="tables" @click="openTable" />
  </main>
</template>

<script setup lang="ts">
  import { onMounted, ref } from 'vue';
  import { useRoute, useRouter } from 'vue-router';
  import { useConnectionState } from '@/stores/connection';
  import EntitiesList from '@/components/entities-list.vue';

  const connection = useConnectionState();
  const router = useRouter();

  const tables = ref<string[]>([]);

  onMounted(async () => {
    const result = await connection.execute<Record<'table_name', string>>(`
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE';
  `);

    tables.value = result.map(it => it.table_name);
  });

  const route = useRoute();

  const openTable = (table: string) =>
    router.push(`/databases/${route.params['database']}/tables/${table}`);
</script>
