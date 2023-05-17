<template>
    <n-modal :show="show">   
        <n-card class="w-64" :title="`Создать запись в ${table}`">
            <n-form>
                <n-form-item v-for="field in fields" :label="field">
                    <n-input @update:value="e => value = { ...value, [field]: e }" :value="value[field]" />
                </n-form-item>
            </n-form>
            <n-button @click="submit">Создать</n-button>
        </n-card>
    </n-modal>
</template>

<script setup lang="ts">
import { defineProps, defineEmits, ref } from 'vue';
import { NModal, NCard, NForm, NFormItem, NInput, NButton } from 'naive-ui';

const value = ref<Record<string, string>>({});

defineProps<{
    show: boolean;
    fields: string[];
    table: string;
}>();

const $emit = defineEmits<{
    (type: 'submit', record: Record<string, string>): void,
    (type: 'update:show', value: boolean): void;
}>();

const submit = () => {
    $emit('update:show', false);
    $emit('submit', value.value);
    value.value = {};
};
</script>