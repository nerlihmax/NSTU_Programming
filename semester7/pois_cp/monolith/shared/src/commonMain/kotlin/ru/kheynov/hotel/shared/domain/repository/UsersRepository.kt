package ru.kheynov.hotel.shared.domain.repository

import ru.kheynov.hotel.shared.domain.entities.RefreshTokenInfo
import ru.kheynov.hotel.shared.domain.entities.User
import ru.kheynov.hotel.shared.domain.entities.UserInfo
import ru.kheynov.hotel.shared.domain.entities.UserUpdate
import ru.kheynov.hotel.shared.jwt.RefreshToken

interface UsersRepository {
    suspend fun registerUser(user: User, passwordHash: String): Boolean
    suspend fun getUserInfoByID(id: String): UserInfo?
    suspend fun getUserByID(id: String): User?
    suspend fun getUserByEmail(email: String): UserInfo?
    suspend fun deleteUserByID(id: String): Boolean
    suspend fun updateUserByID(id: String, update: UserUpdate): Boolean
    suspend fun updateUserRefreshToken(
        userId: String,
        clientId: String,
        newRefreshToken: String,
        refreshTokenExpiration: Long,
    ): Boolean

    suspend fun getRefreshToken(oldRefreshToken: String): RefreshTokenInfo?
    suspend fun getRefreshToken(
        userId: String,
        clientId: String
    ): RefreshTokenInfo?

    suspend fun createRefreshToken(
        userId: String,
        clientId: String,
        refreshToken: RefreshToken
    ): Boolean

}