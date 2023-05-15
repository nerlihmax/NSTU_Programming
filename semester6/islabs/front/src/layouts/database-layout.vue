<template>
  <database-header class="w-full" />
  <div class="p-4 md:p-8 flex flex-grow flex-row gap-4">
    <nav v-if="showTablesList">
      <tables-list
        :tables="tables"
        @click="openTable"
        :active-table="currentTable"
      />
    </nav>
    <router-view class="w-full h-full" />
  </div>
</template>

<script setup lang="ts">
  import { computed, onMounted, ref } from 'vue';
  import { RouterView, useRouter, useRoute } from 'vue-router';
  import { useConnectionState } from '@/stores/connection';
  import DatabaseHeader from '@/components/database-header.vue';
  import TablesList from '@/components/tables-list.vue';

  const router = useRouter();
  const route = useRoute();
  const connection = useConnectionState();

  if (connection.isConnected) {
    router.replace('/database');
  } else {
    router.replace('/authorize');
  }

  const tables = ref<string[]>([]);

  onMounted(async () => {
    const result = await connection.execute<Record<'table_name', string>>(`
    SELECT datname as table_name
    FROM pg_database
    WHERE datistemplate = false;
  `);

    tables.value = result.map(it => it.table_name);
  });

  const showTablesList = computed(() => route.path !== '/database');
  const currentTable = computed(() => route.params['table'] as string);

  const openTable = (table: string) => router.push(`/database/${table}`);
</script>
