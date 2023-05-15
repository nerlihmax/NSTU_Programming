import { client } from '@/core/axios-client';

export const executeQuery = async <T extends Record<string, unknown>>(
  query: string,
): Promise<T[]> => {
  const res = await client.post<Record<string, T>>('/query', query);

  const arr = [];

  for (const key in res.data) {
    arr.push(res.data[key]);
  }

  return arr;
};
