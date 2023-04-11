import { exists } from '@/core/utils/exists'

export const checkConnection = (): boolean =>
  exists(localStorage.getItem('session'))
