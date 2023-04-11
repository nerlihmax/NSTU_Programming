import { client } from '@/core/axios-client'

export const executeQuery = async (
  query: string,
): Promise<boolean | string> => {
  const res = await client.post('/query', query)
  return res.status == 200 ? true : res.data
}
