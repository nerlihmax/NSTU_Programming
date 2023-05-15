<template>
  <n-button @click="saveEdited">Сохранить</n-button>
  <n-data-table :columns="columns" :data="data" />
</template>

<script setup lang="ts">
  import { useRoute } from 'vue-router';
  import { type DataTableColumns, NDataTable, NInput } from 'naive-ui';
  import { useConnectionState } from '@/stores/connection';
  import { computed, createTextVNode, h, ref, watchEffect } from 'vue';

  const route = useRoute();
  const connection = useConnectionState();

  const tableName = computed(() => route.params['table']);

  const columns = ref<DataTableColumns>([]);
  const data = ref<Array<Record<string, unknown>>>([]);

  console.log(route.params);

  const edited = ref<
    {
      id: string;
      changedFields: Record<string, string>;
    }[]
  >([]);

  watchEffect(async () => {
    console.log(tableName.value);
    const cols = await connection.execute<{
      column_name: string;
      data_type: 'integer' | 'text';
    }>(`
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '${tableName.value}';`);

    const datas = await connection.execute(`
    SELECT *
    FROM "${tableName.value}"
  `);

    columns.value = cols.map(({ column_name }) => ({
      key: column_name,
      title: column_name,
      render: (row, index) => {
        const col_data = row[column_name];
        if (typeof col_data !== 'string') return;
        if (column_name === 'id') return createTextVNode(col_data);
        return h(NInput, {
          value: col_data,
          onUpdateValue: value => {
            data.value[index][column_name] = value;
            const found = edited.value?.findIndex(it => row['id'] === it.id);
            if (!found) {
              const key = row['id'];
              if (typeof key !== 'string') return;
              edited.value?.push({
                id: key,
                changedFields: {
                  [column_name]: value,
                },
              });
              return;
            }
            if (!edited.value) return;
            edited.value[found].changedFields[column_name] = value;
          },
        });
      },
    }));

    data.value = datas;

    console.log(columns);
  });

  const saveEdited = async () => {
    // await connection.execute(`
    // UPDATE ${tableName.value}
    // set
    // `);
  };
</script>
