package ru.kheynov.hotel.shared.data.storage

import ru.kheynov.hotel.shared.jwt.TokenPair

expect class TokenStorage {
    fun saveToken(token: TokenPair)
    fun getRefreshToken(): String?
    fun clearToken()
}