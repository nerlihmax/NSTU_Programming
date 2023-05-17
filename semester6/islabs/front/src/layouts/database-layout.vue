<template>
  <database-header class="w-full" />
  <div class="p-4 md:p-8 flex flex-grow flex-row gap-4">
    <div class="flex flex-col space-y-2">
      <n-button
        @click="saveBackupOnClick"
        size="large"
        :type="'success'"
        v-if="route.params['table']"
      >
        Создать бекап
        <template #icon>
          <n-icon :component="ArrowDownload24Regular" />
        </template>
      </n-button>
      <n-button
        @click="restoreBackupOnClick"
        size="large"
        :type="'success'"
        v-if="route.params['table']"
      >
        Восстановить бекап
        <template #icon>
          <n-icon :component="ArrowUpload24Regular" />
        </template>
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
  import { NButton, NIcon, useLoadingBar, useMessage } from 'naive-ui';
  import { saveBackup } from '@/core/use-cases/backups';
  import { ArrowDownload24Regular, ArrowUpload24Regular } from '@vicons/fluent';

  const router = useRouter();
  const route = useRoute();
  const connection = useConnectionState();
  const loadingBar = useLoadingBar();
  const message = useMessage();

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
    loadingBar.start();
    try {
      await saveBackup();
      loadingBar.finish();
      message.success('Бэкап сохранен!');
    } catch {
      message.warning('Не удалось сохранить бэкап!');
      loadingBar.error();
    }
  };

  const restoreBackupOnClick = () => {
    router.push(
      `/databases/${route.params.database}/tables/${route.params.table}/backups`,
    );
  };

  const openTable = (table: string) =>
    router.push(`/databases/${route.params.database}/tables/${table}`);
</script>
