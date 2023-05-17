import { client } from '@/core/axios-client';

export const fetchBackups = async (): Promise<[string]> => {
  const res = await client.get('backups');
  return res.data;
};

export const saveBackup = async (): Promise<void> => await client.post('save');

export const restoreBackup = async (backupName: string): Promise<void> =>
  await client.post('restore', backupName);
