<template>
  <main class="flex flex-col items-center justify-center px-8 md:px-48">
    <tables-list :tables="tables" @click="openTable" />
  </main>
</template>

<script setup lang="ts">
  import { onMounted, ref } from 'vue';
  import { useRouter } from 'vue-router';
  import TablesList from '@/components/tables-list.vue';
  import { useConnectionState } from '@/stores/connection';

  const connection = useConnectionState();
  const router = useRouter();

  const tables = ref<string[]>([]);

  onMounted(async () => {
    const result = await connection.execute<Record<'table_name', string>>(`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema='public'
      AND table_type='BASE TABLE';
    `);

    tables.value = result.map(it => it.table_name);
  });

  const openTable = (table: string) => router.push(`/database/${table}`);
</script>
