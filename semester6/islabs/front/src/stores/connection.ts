import { computed, ref } from 'vue';
import { defineStore } from 'pinia';
import {
  type Connection,
  connectDb,
  disconnectDb,
  getConnection,
} from '@/core/use-cases/db-connection';
import { executeQuery } from '@/core/use-cases/execute-query';

export const useConnectionState = defineStore('connection', () => {
  const connection = ref<null | Connection>(getConnection());

  const isConnected = computed(() => Boolean(connection.value));

  const connect = async (conn: Connection) => {
    const result = await connectDb(conn);

    if (result === true) {
      connection.value = conn;
    }
  };

  const disconnect = async () => {
    if (connection.value) {
      const result = await disconnectDb(connection.value);

      if (result === true) {
        connection.value = null;
      }

      console.log(result);
    } else {
      throw new Error('u trying disconnect while been already disconnected');
    }
  };

  const execute = async <T extends Record<string, unknown>>(query: string) => {
    if (connection.value) {
      const result = await executeQuery<T>(query);

      return result;
    } else {
      throw new Error(
        'u trying execute query while you are not connected, wtf man, chill',
      );
    }
  };

  return {
    connection,
    isConnected,
    connect,
    disconnect,
    execute,
  };
});
