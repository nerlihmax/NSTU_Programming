<template>
  <header
    class="px-4 py-2 md:px-8 md:py-4 shadow-lg flex flex-row justify-between items-center rounded-b-xl gap-2 md:gap-4"
  >
    <section
      class="flex flex-col gap-2 md:flex-row md:items-center md:gap-8 md:w-full"
    >
      <h1 class="text-xl">
        <template v-if="uri">Подключен</template>
        <template v-else>Не подключен</template>
      </h1>
      <template v-if="uri">
        <code class="hidden md:block">{{ uri.long }}</code>
        <code class="md:hidden">{{ uri.short }}</code>
      </template>
    </section>
    <n-button v-if="connection.isConnected" @click="disconnect">
      Отключиться
    </n-button>
  </header>
</template>

<script setup lang="ts">
  import { computed } from 'vue';
  import { useRouter } from 'vue-router';
  import { useConnectionState } from '@/stores/connection';
  import { NButton } from 'naive-ui';

  const connection = useConnectionState();
  const router = useRouter();

  const uri = computed(() => {
    const conn = connection.connection;

    if (!conn) {
      return false;
    }

    return {
      long: `postgresql://${conn.user}:***@${conn.host}:${conn.port}/${conn.database}`,
      short: `${conn.host}:${conn.port}`,
    };
  });

  const disconnect = async () => {
    try {
      await connection.disconnect();
    } catch (error) {
      console.error(error);
    } finally {
      router.push('/');
    }
  };
</script>
