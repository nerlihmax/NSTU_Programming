import axios from 'axios'

const BASE_URL = 'http://localhost:8080/api/'

export const client = axios.create({
  baseURL: BASE_URL,
})

client.interceptors.response.use(async response => {
  console.log('saving session')
  const sessId = response.headers['user_session']
  console.log(sessId)
  console.log(`Session id: ${sessId}`)

  if (typeof sessId === 'string') {
    localStorage.setItem('session', sessId)
  }
  return response
})

client.interceptors.request.use(async request => {
  const sessId = localStorage.getItem('session')
  if (sessId) {
    request.headers.set('user_session', sessId)
  }
  return request
})
