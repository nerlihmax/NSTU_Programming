import { client } from '@/core/axios-client';
import { exists } from '@/utils/exists';

export type Connection = {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
};

export const connectDb = async (
  request: Connection,
): Promise<boolean | string> => {
  const res = await client.post('/connect', request);
  localStorage.setItem('connection', JSON.stringify(request));
  return res.status == 200 ? true : res.data.toString();
};

export const reconnectDb = async () => {
  const request = JSON.parse(localStorage.getItem('connection') ?? '') ?? null;
  if (request) {
    await client.post('/connect', request);
  }
};

export const disconnectDb = async (
  request: Connection,
): Promise<boolean | string> => {
  const res = await client.post('/disconnect', request);
  localStorage.removeItem('connection');
  return res.status == 200 ? true : res.data.toString();
};

export const isSessionOpened = (): boolean =>
  exists(localStorage.getItem('session'));

export const getConnection = (): Connection | null => {
  const stored = localStorage.getItem('connection');
  return stored ? JSON.parse(stored) : null;
};
