<template>
  <main class="flex flex-col items-center justify-center px-8 md:px-48">
    <entities-list :entities="backups" @click="restoreBackupOnClick" />
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import EntitiesList from '@/components/entities-list.vue';
import { fetchBackups, restoreBackup } from '@/core/use-cases/backups';
import { useLoadingBar, useMessage } from 'naive-ui';

const router = useRouter();
const loadingBar = useLoadingBar();
const message = useMessage();

const backups = ref<string[]>([]);

onMounted(async () => {
  backups.value = await fetchBackups();
});

const route = useRoute();

const restoreBackupOnClick = async (backup: string) => {

  loadingBar.start();
  try {
    await restoreBackup(backup);
    loadingBar.finish();
    message.success('Бэкап загружен!');
    await router.push(
      `/databases/${route.params['database']}/tables/${route.params['table']}`,
    );
  } catch {
    message.warning('Не удалось загрузить бэкап!');
    loadingBar.error();
  }
};
</script>
