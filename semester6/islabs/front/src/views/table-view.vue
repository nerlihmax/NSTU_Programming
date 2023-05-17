<template>
  <div class="flex flex-col items-center">
    <n-data-table :columns="columns" :data="data" />
    <n-button @click="addItem" size="large" :type="'success'" class="mt-8">
      Добавить запись
    </n-button>
    <n-button @click="saveEdited" size="large" :type="'success'" class="mt-8">
      Сохранить
    </n-button>
  </div>
</template>

<script setup lang="ts">
  import { useRoute } from 'vue-router';
  import { type DataTableColumns, NButton, NDataTable, NInput } from 'naive-ui';
  import { useConnectionState } from '@/stores/connection';
  import { computed, createTextVNode, h, ref, watch, watchEffect } from 'vue';
  import type { InternalRowData } from 'naive-ui/es/data-table/src/interface';

  const route = useRoute();
  const connection = useConnectionState();

  const tableName = computed(() => route.params['table']);

  const columns = ref<DataTableColumns>([]);
  const data = ref<Array<Record<string, unknown>>>([]);

  const edited = ref<
    {
      id?: string;
      changedFields: Record<string, string>;
      isNew: boolean;
    }[]
  >([]);

  watch(tableName, () => (edited.value = []));

  watchEffect(async () => {
    console.log(tableName.value);
    const cols = await connection.execute<{
      column_name: string;
      data_type: 'integer' | 'text';
    }>(`
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '${tableName.value}';`);

    columns.value = [
      ...cols.map(({ column_name }) => ({
        key: column_name,
        title: column_name,
        render: (row: InternalRowData, index: number) => {
          const col_data = row[column_name];
          if (typeof col_data !== 'string') return;
          if (
            !cols.find(({ column_name }) => column_name === 'id') ||
            column_name === 'id'
          )
            return createTextVNode(col_data);
          return h(NInput, {
            value: col_data,
            onUpdateValue: value => {
              data.value[index][column_name] = value;
              const found = edited.value?.findIndex(it => row['id'] === it.id);
              console.log(found);
              if (found === -1) {
                const key = row['id'];
                if (typeof key !== 'string') return;
                edited.value?.push({
                  isNew: false,
                  id: key,
                  changedFields: {
                    [column_name]: value,
                  },
                });
                return;
              } else {
                edited.value[found].changedFields[column_name] = value;
              }
              if (!edited.value) return;
            },
          });
        },
      })),
      cols.find(({ column_name }) => column_name === 'id')
        ? {
            key: 'D',
            title: 'Delete',
            render: row => {
              return h(NButton, {
                value: 'D',
                onClick: () => {
                  deleteRow(row['id'] as number);
                },
              });
            },
          }
        : [],
    ] as DataTableColumns;

    data.value = await fetchData();

    console.log(columns);
  });

  const addItem = () => {
    edited.value.push({
      isNew: true,
      changedFields: {
        // I have not any fucking idea how to implement this
        ...Object.entries(data.value).map(() => {
          {
          }
        }),
      },
    });
    data.value.push({});
  };

  const fetchData = async () =>
    await connection.execute(`
      SELECT *
      FROM "${tableName.value}"
    `);

  const deleteRow = async (index: number) => {
    if (!index) return;
    await connection.execute(
      `delete
             from ${tableName.value}
             where id = ${index};`,
    );
    data.value = await fetchData();
  };

  const saveEdited = async () => {
    for (const value of edited.value) {
      const insertOperation = `insert into ${tableName.value}
                                     values (${Object.entries(
                                       value.changedFields,
                                     )
                                       .filter(it => it[0] != 'id')
                                       .map(col => col[1])
                                       .join(', ')});`;
      const updateOperation = `update ${tableName.value}
                                     set ${Object.entries(value.changedFields)
                                       .map(col => `${col[0]} = '${col[1]}'`)
                                       .join(', ')}
                                     WHERE id = '${value.id}';`;
      console.log(insertOperation);
      if (value.isNew) await connection.execute(insertOperation);
      else await connection.execute(updateOperation);
    }
  };
</script>
