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
  localStorage.setItem('connection', JSON.stringify(request))
  return res.status == 200 ? true : res.data.toString()
}

export const reconnectDb = async () => {
  const request = JSON.parse(localStorage.getItem('connection') ?? '') ?? null
  if (request) {
    await client.post('/connect', request)
  }
}

export const disconnectDb = async (
  request: ConnectRequest,
): Promise<boolean | string> => {
  const res = await client.post('/disconnect', request)
  localStorage.removeItem('connection')
  return res.status == 200 ? true : res.data.toString()
}
