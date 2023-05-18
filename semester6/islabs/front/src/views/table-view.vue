<template>
  <div class="flex flex-col items-center space-y-4">
    <n-data-table :columns="[...columns, ...deleteCols]" :data="data" />
    <n-button @click="openModal" size="large" :type="'success'">
      Добавить запись
      <template #icon>
        <n-icon :component="Add24Regular" />
      </template>
    </n-button>
    <n-button @click="saveEdited" size="large" :type="'success'">
      Сохранить
      <template #icon>
        <n-icon :component="Save24Regular" />
      </template>
    </n-button>
    <create-record-modal
      :fields="columns.map(col => 'key' in col ? col.key as string : '').filter(col => col !== 'id')"
      :table="tableName"
      v-model:show="showCreateModal"
      @submit="createRecord"
    />
  </div>
</template>

<script setup lang="ts">
  import { useRoute } from 'vue-router';
  import { computed, createTextVNode, h, ref, watch, watchEffect } from 'vue';
  import {
    type DataTableColumns,
    NButton,
    NDataTable,
    NIcon,
    NInput,
    useLoadingBar,
    useMessage,
  } from 'naive-ui';
  import type { InternalRowData } from 'naive-ui/es/data-table/src/interface';
  import { Add24Regular, Delete24Regular, Save24Regular } from '@vicons/fluent';
  import { useConnectionState } from '@/stores/connection';
  import CreateRecordModal from '@/components/create-record-modal.vue';

  const route = useRoute();
  const connection = useConnectionState();

  const loadingBar = useLoadingBar();

  const message = useMessage();

  const tableName = computed(() => route.params['table'] as string);

  const columns = ref<DataTableColumns>([]);
  const data = ref<Array<Record<string, unknown>>>([]);
  const showCreateModal = ref(false);

  watchEffect(() => {
    console.log(columns.value);
  });

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
    ] as DataTableColumns;

    data.value = await fetchData();

    console.log(columns);
  });

  const deleteCols = computed<DataTableColumns>(() =>
    columns.value.find(column => 'key' in column && column.key === 'id')
      ? [
          {
            key: 'D',
            title: 'Delete',
            render: row => {
              return h(
                NButton,
                {
                  value: 'D',
                  onClick: () => {
                    deleteRow(row['id'] as number);
                  },
                  type: 'error',
                  ghost: true,
                },
                [h(NIcon, { component: Delete24Regular })],
              );
            },
          },
        ]
      : [],
  );

  const openModal = () => {
    showCreateModal.value = true;
  };

  const fetchData = async () => {
    if (!tableName.value) return;
    try {
      loadingBar.start();
      const res = await connection.execute(`
      SELECT *
      FROM "${tableName.value}"
    `);
      loadingBar.finish();
      return res;
    } catch {
      loadingBar.error();
      message.error('Произошла ошибка при получении данных');
      return;
    }
  };

  const deleteRow = async (index: number) => {
    if (!index) return;
    try {
      loadingBar.start();
      await connection.execute(
        `delete
       from ${tableName.value}
       where id = ${index};`,
      );
      loadingBar.finish();
      message.success('Запись удалена!1!');
    } catch {
      loadingBar.error();
      message.error('Произошло недоразумение при удалении!1!');
    }
    data.value = await fetchData();
  };

  const createRecord = async (record: Record<string, string>) => {
    const entries = Object.entries(record).filter(([field]) => field !== 'id');
    loadingBar.start();
    try {
      await connection.execute(`
      insert into ${tableName.value}
        (${entries.map(([field]) => field).join(',')})
      values (${entries.map(([_, value]) => `'${value}'`).join(',')});
    `);
      loadingBar.finish();
      message.success('Сохранено!1!');
    } catch {
      loadingBar.error();
      message.error('Произошло недоразумение при добавлении записи!1!');
    }
    data.value = await fetchData();
  };

  const saveEdited = async () => {
    for (const value of edited.value) {
      const insertOperation = `insert into ${tableName.value}
                             values (${Object.entries(value.changedFields)
                               .filter(it => it[0] != 'id')
                               .map(col => col[1])
                               .join(', ')});`;
      const updateOperation = `update ${tableName.value}
                             set ${Object.entries(value.changedFields)
                               .map(col => `${col[0]} = '${col[1]}'`)
                               .join(', ')}
                             WHERE id = '${value.id}';`;
      console.log(insertOperation);
      loadingBar.start();
      try {
        if (value.isNew) await connection.execute(insertOperation);
        else await connection.execute(updateOperation);
        loadingBar.finish();
        message.success('Сохранено!1!');
      } catch {
        loadingBar.error();
        message.error('Произошло недоразумение при сохранении!1!');
      }
    }
  };
</script>
