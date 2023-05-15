<template>
  <n-card size="small">
    <n-form
      class="mb-4"
      :rules="validationRules"
      :model="formValue"
      ref="formRef"
    >
      <n-form-item path="host" label="Хост">
        <n-input
          placeholder="Введите хост"
          v-model:value="formValue.host"
          :disabled="$props.disabled"
        />
      </n-form-item>
      <n-form-item path="port" label="Порт">
        <n-input
          placeholder="Введите порт"
          @update:value="e => (formValue.port = Number(e))"
          :disabled="$props.disabled"
        />
      </n-form-item>
      <n-form-item path="user" label="Пользователь">
        <n-input
          placeholder="Введите пользователя"
          v-model:value="formValue.user"
          :disabled="$props.disabled"
        />
      </n-form-item>
      <n-form-item path="password" label="Пароль">
        <n-input
          placeholder="Введите пароль"
          type="password"
          v-model:value="formValue.password"
          :disabled="$props.disabled"
        />
      </n-form-item>
    </n-form>
    <n-button type="default" @click="submit" :disabled="$props.disabled">
      Подключиться
    </n-button>
  </n-card>
</template>

<script setup lang="ts">
  type FormValue = {
    host?: string;
    port?: number;
    user?: string;
    password?: string;
  };

  export type FormSubmitValue = Required<FormValue> & { database: string };

  import { ref } from 'vue';
  import type { FormInst, FormRules } from 'naive-ui';
  import { NButton, NCard, NForm, NFormItem, NInput } from 'naive-ui';

  const $props = defineProps<{ disabled?: boolean }>();

  const $emit =
    defineEmits<(event: 'submit', value: FormSubmitValue) => void>();

  const formValue = ref<FormValue>({});
  const formRef = ref<FormInst | null>(null);

  const validationRules: FormRules = {
    host: {
      required: true,
      message: 'Пожалуйста, введите хост',
      trigger: ['blur'],
    },
    port: {
      required: true,
      type: 'number',
      message: 'Пожалуйста, введите порт',
      trigger: ['blur'],
    },
    user: {
      required: true,
      message: 'Пожалуйста, введите пользователя',
      trigger: ['blur'],
    },
    password: {
      required: true,
      message: 'Пожалуйста, введите пароль',
      trigger: ['blur'],
    },
  };

  const submit = (e: MouseEvent) => {
    e.preventDefault();
    formRef.value?.validate(errors => {
      if (!errors) {
        $emit('submit', {
          ...formValue.value,
          database: formValue.value.user,
        } as FormSubmitValue);
      }
    });
  };
</script>
