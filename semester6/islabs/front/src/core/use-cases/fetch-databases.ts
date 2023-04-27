import { client } from '@/core/axios-client';

export const fetchDatabases = async (): Promise<[string]> => {
  const res = await client.get('databases');
  return res.data;
};
