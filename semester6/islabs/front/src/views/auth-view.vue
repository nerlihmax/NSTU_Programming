<template>
  <main class="flex flex-col items-center justify-center gap-4 md:gap-8">
    <h1 class="text-2xl">Подключиться</h1>
    <credentials-form
      class="w-10/12 md:w-6/12"
      @submit="submit"
      :disabled="loading"
    />
  </main>
</template>

<script setup lang="ts">
  import { ref } from 'vue';
  import { useRouter } from 'vue-router';
  import { useConnectionState } from '@/stores/connection';
  import CredentialsForm, {
    type FormSubmitValue,
  } from '@/components/credentials-form.vue';

  const router = useRouter();
  const connection = useConnectionState();

  if (connection.isConnected) {
    router.replace('/databases');
  }

  const loading = ref<boolean>(false);

  const submit = async (value: FormSubmitValue) => {
    loading.value = true;
    await connection.connect(value);
    loading.value = false;
    router.push('/databases');
  };
</script>
