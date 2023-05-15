<template>
  <n-data-table :columns="columns" :data="data" />
</template>

<script setup lang="ts">
  import { useRoute } from 'vue-router';
  import { type DataTableColumns, NCard, NDataTable } from 'naive-ui';
  import { useConnectionState } from '@/stores/connection';
  import { onMounted, ref } from 'vue';

  const route = useRoute();
  const connection = useConnectionState();

  const tableName = route.params['table'];

  const columns = ref<DataTableColumns>([]);
  const data = ref<Array<Record<string, unknown>>>([]);

  onMounted(async () => {
    const cols = await connection.execute<{
      column_name: string;
      data_type: 'integer' | 'text';
    }>(`
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name = 'my_table'
    `);

    const datas = await connection.execute(`
      SELECT * FROM my_table
    `);

    columns.value = cols.map(({ column_name, data_type }) => ({
      key: column_name,
      title: column_name,
    }));

    data.value = datas;

    console.log(columns);
  });
</script>
