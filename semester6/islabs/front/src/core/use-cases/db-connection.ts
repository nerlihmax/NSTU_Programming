import { client } from '@/core/axios-client'

export type ConnectRequest = {
  host: string
  port: string
  user: string
  password: string
  database: string
}

export const connectDb = async (
  request: ConnectRequest,
): Promise<boolean | string> => {
  const res = await client.post('/connect', request)
  return res.status == 200 ? true : res.data.toString()
}

export const disconnectDb = async (
  request: ConnectRequest,
): Promise<boolean | string> => {
  const res = await client.post('/disconnect', request)
  return res.status == 200 ? true : res.data.toString()
}
