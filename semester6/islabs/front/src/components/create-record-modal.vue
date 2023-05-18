<template>
  <n-modal v-model:show="show">
    <n-card class="w-64" :title="`Создать запись в ${table}`">
      <n-form>
        <n-form-item v-for="field in fields" :label="field">
          <n-input
            @update:value="e => (value = { ...value, [field]: e })"
            :value="value[field]"
          />
        </n-form-item>
      </n-form>
      <n-button @click="submit">Создать</n-button>
    </n-card>
  </n-modal>
</template>

<script setup lang="ts">
  import { defineEmits, defineProps, ref, watchEffect } from 'vue';
  import { NButton, NCard, NForm, NFormItem, NInput, NModal } from 'naive-ui';

  const value = ref<Record<string, string>>({});

  const show = ref(false);

  const props = defineProps<{
    show: boolean;
    fields: string[];
    table: string;
  }>();

  watchEffect(() => {
    show.value = props.show;
  });

  const $emit = defineEmits<{
    (type: 'submit', record: Record<string, string>): void;
    (type: 'update:show', value: boolean): void;
  }>();

  watchEffect(() => {
    $emit('update:show', show.value);
  });

  const close = () => {
    $emit('update:show', false);
  };

  const submit = () => {
    $emit('update:show', false);
    $emit('submit', value.value);
    value.value = {};
  };
</script>
