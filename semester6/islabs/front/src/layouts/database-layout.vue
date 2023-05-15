<template>
  <database-header class="w-full" />
  <div class="p-4 md:p-8 flex flex-grow flex-row gap-4">
    <nav v-if="shouldShowTablesList">
      <entities-list
        :entities="tables"
        @click="openTable"
        :active-entity="currentTable"
      />
    </nav>
    <router-view class="w-full h-full" />
  </div>
</template>

<script setup lang="ts">
  import { computed, ref, watchEffect } from 'vue';
  import { RouterView, useRoute, useRouter } from 'vue-router';
  import { useConnectionState } from '@/stores/connection';
  import DatabaseHeader from '@/components/database-header.vue';
  import EntitiesList from '@/components/entities-list.vue';

  const router = useRouter();
  const route = useRoute();
  const connection = useConnectionState();

  if (connection.isConnected) {
    router.replace('/databases');
  } else {
    router.replace('/authorize');
  }

  const tables = ref<string[]>([]);

  watchEffect(async () => {
    if (!connection.isConnected) return;
    const result = await connection.execute<Record<'table_name', string>>(`
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE';
  `);

    console.log(result);

    tables.value = result.map(it => it.table_name);
  });

  const shouldShowTablesList = computed(
    () => !!route.params.database && !!route.params.table,
  );
  const currentTable = computed(() => route.params['table'] as string);

  const openTable = (table: string) =>
    router.push(`/databases/${route.params.database}/tables/${table}`);
</script>
