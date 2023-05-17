<template>
  <database-header class="w-full" />
  <div class="p-4 md:p-8 flex flex-grow flex-row gap-4">
    <div class="flex flex-col">
      <n-button
        @click="saveBackupOnClick"
        size="large"
        :type="'success'"
        class="mt-4 mb-2"
        v-if="route.params['table']"
      >
        Создать бекап
      </n-button>
      <n-button
        @click="restoreBackupOnClick"
        size="large"
        :type="'success'"
        class="mb-4"
        v-if="route.params['table']"
      >
        Восстановить бекап
      </n-button>
      <nav v-if="shouldShowTablesList">
        <entities-list
          :entities="tables"
          @click="openTable"
          :active-entity="currentTable"
        />
      </nav>
    </div>
    <router-view class="w-full h-full" />
  </div>
</template>

<script setup lang="ts">
  import { computed, ref, watchEffect } from 'vue';
  import { RouterView, useRoute, useRouter } from 'vue-router';
  import { useConnectionState } from '@/stores/connection';
  import DatabaseHeader from '@/components/database-header.vue';
  import EntitiesList from '@/components/entities-list.vue';
  import { NButton } from 'naive-ui';
  import { saveBackup } from '@/core/use-cases/backups';

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

  const saveBackupOnClick = async () => {
    await saveBackup();
    alert('Backup saved!');
  };

  const restoreBackupOnClick = () => {
    router.push(
      `/databases/${route.params.database}/tables/${route.params.table}/backups`,
    );
  };

  const openTable = (table: string) =>
    router.push(`/databases/${route.params.database}/tables/${table}`);
</script>
