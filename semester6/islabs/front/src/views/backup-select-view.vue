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

  const router = useRouter();

  const backups = ref<string[]>([]);

  onMounted(async () => {
    backups.value = await fetchBackups();
  });

  const route = useRoute();

  const restoreBackupOnClick = async (backup: string) => {
    console.log('loading backup...');
    await restoreBackup(backup);
    alert('Backup loaded');
    await router.push(
      `/databases/${route.params['database']}/tables/${route.params['table']}`,
    );
  };
</script>
